# ExRec: 教育推荐多智能体框架

带反思机制的教育推荐系统,针对学生薄弱知识点(Knowledge Component, KC)生成个性化补弱推荐。

**核心 claim**: *结构化反思 + 置信度演化 + 经验库自维护* 三件套,让推荐策略可以在推理过程中持续自我修正,而无需重新训练任何模型。

---

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GlobalAgent (调度+反思维护)                     │
│                                                                         │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│   │ TriggerJudge │ → │ ReflectionEngine  │ → │   ExperienceBank    │   │
│   │ (阈值触发)    │    │ (LLM生成结构化反思)│    │ (检索/置信度/合并)   │   │
│   └──────────────┘    └──────────────────┘    └─────────────────────┘   │
│                              ↑   ↓                                     │
└──────────────────────────────│───│───────────────────────────────────────┘
                               │   │ retrieve_experiences()
            process_round()    │   │
                               │   ↓
   ┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │ StudentAgent  │ →  │   TeacherAgent   │ →  │  StudentAgent   │
   │ build_profile │    │ recommend(K)     │    │ simulate(qids)  │
   │ (DKT诊断+画像) │    │ (3级检索+反思应用) │    │ (DKT预测+采样)   │
   └───────────────┘    └──────────────────┘    └─────────────────┘
        ↑                       ↑                       │
        │ KC mastery            │ candidates            │
        │                       │                       ↓
    ┌────────────────────────────────────────────────────────┐
    │           data_processing (题目库/嵌入/序列)             │
    │   QuestionBank   EmbeddingStore   InteractionLoader    │
    └────────────────────────────────────────────────────────┘
```

每个学生的完整推理闭环:

```
build_profile   → retrieve_experiences → recommend → simulate → process_round
(学生Agent)         (全局Agent)         (教师Agent) (学生Agent)   (全局Agent)
```

---

## 2. 项目结构

```
ExRec/
├── configs/                        ← YAML 配置
│   ├── default.yaml                  主配置(数据路径/LLM/agent/反思/评估)
│   ├── ablation_no_reflection.yaml   消融:关闭整个反思
│   ├── ablation_flat_reflection.yaml 消融:不使用结构化 action_delta
│   └── ablation_no_confidence.yaml   消融:反思永久有效不衰减
│
├── utils/                          ← 公共工具
│   ├── config_loader.py              YAML 加载 + _base_ 继承 + 环境变量插值
│   ├── logger.py                     文件+控制台日志 + JsonlLogger 结构化追踪
│   ├── seed.py                       统一随机种子
│   └── embedding_utils.py            余弦相似度/批量检索/归一化
│
├── llm_adapter/                    ← LLM 后端适配
│   ├── base.py                       BaseLLMAdapter / LLMResponse
│   ├── openai_compat_adapter.py      OpenAI兼容协议(含 vLLM 本地)
│   ├── deepseek_adapter.py           DeepSeek 平台
│   ├── modelscope_adapter.py         ModelScope API-Inference
│   ├── mock_adapter.py               Mock 后端(单测/离线调试)
│   ├── cache.py                      基于 hash 的 LLM 调用缓存
│   └── factory.py                    LLMFactory.from_config(cfg) 统一入口
│
├── data_processing/                ← 数据加载
│   ├── question_bank.py              题目元数据 + KC/难度倒排索引
│   ├── interaction_loader.py         pykt 风格 CSV 序列加载
│   └── embedding_store.py            题目内容/ID/KC 三类嵌入的统一查询
│
├── student_agent/                  ← 学生 Agent
│   ├── student_profile.py            StudentProfile dataclass(画像)
│   ├── dkt_wrapper.py                pykt 模型包装(诊断 + 模拟)
│   ├── profile_generator.py          LLM 生成自然语言画像
│   ├── group_prior.py                GMM 群体先验 + α 融合
│   └── student_agent.py              主类:build_profile() / simulate()
│
├── teacher_agent/                  ← 教师 Agent
│   ├── retrieval_pipeline.py         三级检索(L1结构化→L2向量→L3 LLM)
│   ├── strategy_aggregator.py        多反思 action_delta 聚合
│   └── teacher_agent.py              主类:recommend(profile, experiences, K)
│
├── reflection/                     ← 反思机制(核心)
│   ├── experience_unit.py            ExperienceUnit dataclass + 标签封闭集
│   ├── experience_bank.py            CRUD + 检索 + 置信度演化 + 合并 + 持久化
│   ├── trigger_judge.py              阈值触发判定(severity: high/medium/low/none)
│   └── reflection_engine.py          LLM 结构化反思生成 + 严格 JSON 校验
│
├── global_agent/                   ← 全局 Agent
│   └── global_agent.py               协调 + retrieve_experiences + process_round
│
├── evaluation/                     ← 评估
│   ├── metrics.py                    KC-Coverage / EKG / Diff-Match / Diversity / NDCG / F1 / Hit
│   └── evaluator.py                  按 K_list 批量评估并聚合
│
├── scripts/                        ← 运行脚本
│   ├── run_pipeline.py               主入口:完整推理闭环
│   ├── train_group_prior.py          独立训练 GMM 群体先验
│   └── run_all_ablations.py          顺序跑所有消融
│
├── tests/                          ← 单元测试 (pytest)
│   ├── conftest.py                   公共 fixture
│   ├── test_llm_adapter.py
│   ├── test_experience_bank.py
│   ├── test_reflection_engine.py
│   └── test_metrics.py
│
├── logs/                           ← 运行日志(自动生成)
├── cache/                          ← LLM 缓存(自动生成)
├── experience_bank_storage/        ← 经验库持久化(自动生成)
├── requirements.txt
└── README.md
```

---

## 3. 各模块详解

### 3.1 `utils/` — 公共工具

- **`config_loader.load_config(path)`**
  加载 YAML,支持 `_base_: "default.yaml"` 继承(子配置覆盖父配置),支持 `${ENV:VAR_NAME}` 从环境变量读取。
- **`logger.setup_logging(log_dir, run_name)`** + **`get_logger(name)`**
  统一日志体系:控制台 + 文件双输出。
- **`logger.JsonlLogger(path)`**
  结构化 JSONL 追踪日志,记录每次推荐的输入/输出/反思,便于 pandas 二次分析。
- **`seed.set_seed(seed)`**
  统一 random / numpy / torch 种子。
- **`embedding_utils`**
  `load_embedding_json` / `cosine_similarity` / `batch_cosine` / `normalize`。

### 3.2 `llm_adapter/` — LLM 调用统一入口

通过工厂模式屏蔽底层差异。业务代码只关心:

```python
llm = LLMFactory.from_config(cfg)               # 自动选后端
resp = llm.chat([{"role": "user", "content": "..."}],
                response_format={"type": "json_object"})
```

- 4 个后端:`openai_compat` / `deepseek` / `modelscope` / `mock`
- DeepSeek 和 ModelScope 都继承 OpenAICompatAdapter,仅 `base_url` 和默认模型不同
- 自动重试(默认 3 次,指数退避)
- 自动缓存:基于 `(backend, model, messages, temperature, max_tokens, response_format)` 的 sha256
- 累计 token 与 cache 命中统计

### 3.3 `data_processing/` — 数据加载层

- **`QuestionBank`** 加载 `high_school_annotated_clean.json` + `qid_to_int.json`,构建:
  - `ques_id → 题目字典`
  - `KC → [ques_id]`(同时合并 `knowledge_concepts_list` 与 `kc_mapping_gpt4o.list_KCs`,后者括号注释会被剥离)
  - `难度名 → [ques_id]`
  - `qid_to_int / int_to_qid`
- **`InteractionLoader`** 加载 pykt 风格 CSV:
  - 自动过滤 `-1` 填充
  - `concepts` 字段用 `_` 拆多 KC、用 `,` 拆多 step
  - 允许同学生多行,自动按出现顺序拼接
- **`EmbeddingStore`** 统一三类嵌入:
  - `qid → 内容嵌入`(检索用)
  - `qid → int 嵌入`(DKT 配套)
  - `KC → 嵌入`(薄弱 KC 表征)
  - 提供 `search_questions_by_content(query, candidate_qids, top_n)` 做候选池上的向量检索

### 3.4 `student_agent/` — 学生 Agent

- **`StudentProfile`** dataclass(画像),包含:
  - DKT 诊断:`kc_mastery` / `weak_kcs` / `ability_level` / `avg_mastery`
  - 序列统计:`interaction_count` / `correct_rate` / `interaction_density` / `answered_qids`
  - LLM 生成:`weak_kcs_summary` / `ability_summary` / `preference_summary` / `learning_preference`
  - 表征向量:`individual_embedding` / `group_embedding` / `fused_embedding` / `sig_embedding`(反思检索用)
- **`DKTSimulator`**
  - `diagnose(questions, concepts, responses) → {KC: mastery}`
  - `simulate(history, next_questions) → [P(correct)]`
  - `sample_outcomes(probs) → [0/1]`
  - pykt 不可用时 fallback 到 "KC 正确率 + Laplace 平滑" 的统计方法
- **`ProfileGenerator`** — LLM 驱动,严格 JSON 输出,解析失败 fallback 到模板
- **`GroupPriorModule`** — sklearn GMM(降级 KMeans)
  - `fit(X)` 训练
  - `fuse(individual, n_itx) → (fused, group, α)`,α 随交互次数自适应
- **`StudentAgent.build_profile(sid)`** 流程: DKT 诊断 → 薄弱 KC → 全局统计 → LLM 画像 → 个体表征 → 群体融合 → sig_embedding

### 3.5 `teacher_agent/` — 教师 Agent

三级检索流水线:

- **L1: 结构化召回**(毫秒)— 按 `weak_kcs ∩ difficulty_window` 硬过滤,候选不足时放宽难度
- **L2: 向量精排**(百毫秒)— 双通道余弦:学生融合表征 + 薄弱 KC 表征,加权融合
- **L3: LLM 重排序**(秒)— 给候选 brief 让 LLM 输出 selected_indices + predicted_correct_rates + rationale + strategy_label,LLM 失败时降级 L2 top-K

`StrategyAggregator`:把多条反思的 `action_delta` 按 `confidence` 加权聚合 →
- `difficulty_shift`: 加权平均后四舍五入到 [-2, 2]
- `prereq_check`: 加权投票 ≥ 0.5 → True
- `diversity_target`: 加权平均
- `kc_focus_change`: 并集去重(扩充而非替换)

`TeacherAgent.recommend()` 完整流程:`base_strategy → apply_delta → L1 → L2 → L3 → 统计输出`,输出含 `expected_kg`(预期学习增益)。

### 3.6 `reflection/` — 反思机制(核心)

**`ExperienceUnit`**: 反思的原子单元,包含:
- `student_signature` + `student_sig_embedding`(供检索)
- `action`(本次推荐做了什么)
- `outcome`(预测 vs 实际)
- `lesson` 自然语言总结 + `lesson_tags`(8 项封闭集合)
- `suggested_action_delta`(结构化策略修正)
- `confidence` / `support_count` / `contradict_count` / `status`(active/dormant/retired)/ `severity`

**封闭标签集合**:
```
overestimate_difficulty / underestimate_difficulty
ignore_prerequisite / low_diversity / preference_mismatch
weak_kc_misidentified / redundant_recommendation / other
```

**`ExperienceBank`**: 线程安全的反思库
- `retrieve(query_emb, top_k, min_conf, sim_thr)` — 基于签名嵌入向量检索(自动维度对齐)
- `retrieve_recent(top_k)` — 消融用,不基于相似度
- `update_confidence(exp_id, "support"/"contradict")` — 贝叶斯 Beta 后验 + 时间衰减
- `consolidate(...)` — 贪心聚类合并(规则合并,可注入 `llm_merge_fn` 替换为 LLM 合并)
- `gc()` / `save(path)` / `load(path)` / `stats()`

**`TriggerJudge.judge(probs, sims, kg_expected, kg_actual)`**:
基于 MAE 阈值 + KG gap + 系统性偏差(全错全对)判定 severity:
- `high`: MAE ≥ 0.40 / KG gap ≥ 0.15 / 系统性偏差
- `medium`: MAE ≥ 0.25 / KG gap ≥ 0.08
- `low`: MAE ≥ 0.15
- `none`: 无需反思

**`ReflectionEngine.reflect(...)`**:
调 LLM 生成结构化反思,严格 JSON 解析 + 字段校验:
- `difficulty_shift` clip 到 [-2, 2]
- `tags` 必须在封闭集合内(过滤非法 tag)
- `self_confidence` clip 到 [0, 1]
- 解析失败返回 None(不阻塞推荐链路)

### 3.7 `global_agent/` — 全局 Agent

**对外只有两个核心方法**:

1. **`retrieve_experiences(profile) → List[ExperienceUnit]`** — 供教师 Agent 调用
   - 正常: 基于 `sig_embedding` 检索
   - 消融 `retrieval=False`: 走 `retrieve_recent`

2. **`process_round(profile, recommendation, sim_results, applied_exps, round_id)`** — 推荐结束后调
   - 触发判断 → 反思生成 → 置信度回填:
     - 推荐成功 → 所有 applied_exp 计 `support`
     - `severity == high` → 所有 applied_exp 计 `contradict`
   - 周期性 `consolidate + gc + autosave`

消融开关读自 `cfg.reflection.ablation`:`structured_delta` / `confidence_update` / `retrieval` / `consolidation`。

### 3.8 `evaluation/` — 评估

**主要指标**(对补弱推荐场景适配):
- `KC-Coverage@K` — 推荐题覆盖了多少薄弱 KC
- `Expected-KG@K` — 预期学习增益:`mean(P_correct × Σ_{kc∈weak∩Q_kcs}(1 - mastery_kc))`
- `Difficulty-Match@K` — 推荐难度落入 ZPD 区间(能力档相关)的比例
- `Diversity@K` — 涉及的不同 KC 数 / 题数

**参考指标**(给审稿人对照):NDCG / F1 / Hit @K 的题目级和 KC 级

**反思系统观测指标**(由 `GlobalAgent.stats()` 提供):
- 反思命中率: 检索到经验的次数 / 总轮数
- 反思应用率: 实际使用经验的推荐占比
- 经验库稳定性: active/dormant/retired 分布
- 平均经验置信度

`Evaluator.add_round(...)` 累计每轮,`summarize()` 跨学生取均值与标准差。

---

## 4. 模块间数据通信

每个学生的完整推理闭环:

| Step | 调用 | 输入 | 输出 |
|---|---|---|---|
| 1 | `StudentAgent.build_profile(sid)` | sid + 交互序列(test集时序切片前半段) | `StudentProfile` (含 `weak_kcs`, `kc_mastery`, `sig_embedding`, `fused_embedding`) |
| 2 | `GlobalAgent.retrieve_experiences(profile)` | `profile.sig_embedding` | `List[ExperienceUnit]` (top_k 相关历史经验) |
| 3 | `TeacherAgent.recommend(profile, exps, K)` | `profile` + 反思 + K | `recommendation` 字典(`questions`, `predicted_correct_rates`, `expected_kg`, `kc_focus`, `strategy_label`, `applied_experience_ids` …) |
| 4 | `StudentAgent.simulate(sid, qids)` | sid + 推荐题列表 | `{predicted_probs, simulated_correct, kg_before, kg_after}` |
| 5 | `GlobalAgent.process_round(...)` | profile + recommendation + sim + applied_exps | `(new_exp_or_None, severity, debug)` 并副作用地更新经验库 |
| 6 | `Evaluator.add_round(...)` | profile + recommendation + ground_truth | 累计指标 |

**反思机制的数据闭环**:

```
推荐时:   GlobalAgent.retrieve_experiences()
            → ExperienceBank.retrieve(sig_emb)
            → top_k 经验
            → StrategyAggregator.aggregate(exps) [TeacherAgent 内部]
            → 策略 delta
            → 影响 L1 召回 + L3 重排

推荐后:   GlobalAgent.process_round()
            → TriggerJudge.judge() 判定 severity
            → 若需反思:ReflectionEngine.reflect() 生成 ExperienceUnit
            → ExperienceBank.add()
            → 对每条 applied_exp:ExperienceBank.update_confidence("support" 或 "contradict")
            → 每 N 轮 ExperienceBank.consolidate() + gc()
            → 周期 ExperienceBank.save(path)
```

---

## 5. 如何运行

### 5.1 准备环境

```bash
# 1) 安装 pykt-toolkit(请遵循官方文档)
# https://github.com/pykt-team/pykt-toolkit

# 2) 项目依赖
pip install -r requirements.txt

# 3) 设置 LLM API Key(根据 default.yaml 中的 default_backend)
export DEEPSEEK_API_KEY="sk-..."
# 或 OpenAI / ModelScope
export OPENAI_API_KEY="..."
export MODELSCOPE_API_KEY="..."
```

### 5.2 校验环境(可选,推荐先做)

```bash
# 1) 单元测试(用 Mock LLM,不需要任何外部依赖)
python -m pytest tests/ -v

# 2) 用 Mock LLM 跑一遍最小流水线,确认数据通路通畅
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --backend mock \
    --max-students 5
```

成功的话,会在 `logs/eval_results/` 看到 `default_run_summary.json` 和 `default_run_rounds.json`。

### 5.3 训练群体先验(可选,首次运行)

```bash
python scripts/train_group_prior.py \
    --config configs/default.yaml \
    --sample 1000
```

会输出 `logs/group_prior.pkl`,之后 `run_pipeline.py` 自动加载。
不显式训练也可以,`run_pipeline.py` 检测到没有时会即时拟合 500 个训练学生。

### 5.4 跑主实验

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

主要产出:
- `logs/default_run_<timestamp>.log` — 日志
- `logs/trace_default_run.jsonl` — 每轮详细 trace(JSONL)
- `logs/eval_results/default_run_rounds.json` — 每学生指标
- `logs/eval_results/default_run_summary.json` — 聚合报告
- `experience_bank_storage/exp_bank.pkl` — 反思经验库(可跨实验复用,做消融前注意清空)

### 5.5 跑消融实验

```bash
# 单个消融
python scripts/run_pipeline.py --config configs/ablation_no_reflection.yaml

# 一键跑全部 4 个配置
python scripts/run_all_ablations.py
```

`run_all_ablations.py` 会在每个实验开始前清空 `experience_bank_storage/exp_bank.pkl`,
确保各组结果互相独立。

### 5.6 常用快捷参数

```bash
# 用更小的 max_students 快速调试
python scripts/run_pipeline.py --config configs/default.yaml --max-students 20

# 临时换 LLM 后端
python scripts/run_pipeline.py --config configs/default.yaml --backend deepseek
```

---

## 6. 常见问题

**Q: pykt 模型加载失败/找不到 checkpoint?**
DKTSimulator 自动 fallback 到 "KC 正确率 + Laplace 平滑" 的统计诊断,日志会有 WARNING。这不会影响流水线运行,但 KG 数值会偏离真值。请确认 `cfg.dkt.model_dir` 指向你的预训练目录,且其中含 `config.json` 与 `pretrained_model.ckpt`。

**Q: 嵌入维度不匹配怎么办?**
EmbeddingStore 和 ExperienceBank 都做了自动维度对齐(截断/补 0)。若发现整体维度变化(例如换嵌入模型),建议清空 `experience_bank_storage/` 并重新训练 group_prior,确保表征空间一致。

**Q: 消融实验之间为何要清空经验库?**
经验库是状态化的,若 `ablation_no_reflection` 跑完后留下经验,会污染下一组 `ablation_flat_reflection` 的检索结果。`run_all_ablations.py` 已自动处理,手动跑单个消融时注意自行清空。

**Q: 怎么把 mock 输出换成真实可读的反思?**
直接改 `cfg.llm.default_backend` 为 `deepseek` / `openai_compat` / `modelscope`,设置对应 API Key 环境变量。Mock 后端只用于单测和无网调试。

**Q: 想加新的反思 tag?**
编辑 `reflection/experience_unit.py` 的 `LESSON_TAGS`,然后同步更新 `reflection/reflection_engine.py` 的 `REFLECTION_SYSTEM_PROMPT` 里枚举的封闭集合。`_parse_and_validate` 会自动过滤不在集合中的 tag。

**Q: 想换 DKT 为 AKT / DKVMN / SAKT?**
在 `cfg.dkt.model_type` 改名,`DKTSimulator._init_pykt_model` 会按 `config.model_name` 重建模型。具体的 forward 接口可能略有差异,需要在 `_simulate_with_pykt` 内适配(目前是占位实现,fallback 到统计法)。

---

## 7. 论文 claim 与实验设计的对应关系

| Claim | 对应实现 | 对应消融 |
|---|---|---|
| **结构化反思** 比文本反思更利于策略修正 | `ReflectionEngine` 生成 `action_delta` + `StrategyAggregator` 加权聚合 | `ablation_flat_reflection.yaml` (`structured_delta=false`) |
| **置信度演化** 让错经验自动失效 | `ExperienceBank.update_confidence` Bayesian 后验 + 时间衰减 + 状态机 | `ablation_no_confidence.yaml` (`confidence_update=false`) |
| **检索式反思** 比无差别使用更精准 | `ExperienceBank.retrieve` 基于 sig_embedding | 全反思 vs 无反思 (`ablation_no_reflection.yaml`) |
| **经验合并** 防止经验库无限膨胀 | `ExperienceBank.consolidate` 贪心聚类 + tag 重叠校验 | 比较开关 `consolidation` 前后的经验库稳定性 |

各组合实验跑完后,直接对比 `logs/eval_results/*_summary.json` 里的:
- `kc_coverage@K_mean` / `expected_kg@K_mean` / `difficulty_match@K_mean` — 推荐质量
- `global_agent.experience_bank.active` — 经验库稳定性
- `reflection_apply_rate` — 反思应用率

---

## 8. 下一步可扩展点

- 接入真实 pykt 的 forward 推理(目前 `_simulate_with_pykt` 是占位,可降级到统计法)
- 反思合并 `llm_merge_fn` 改用 LLM 而非规则
- 把学生个体表征换成 Transformer 编码而非 KC 掌握度向量
- 引入异步反思(后台进程批量反思,不阻塞主推荐)
