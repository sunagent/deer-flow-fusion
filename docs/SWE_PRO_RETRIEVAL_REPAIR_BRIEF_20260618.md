# SWE Pro Full-Repo Retrieval 专项修复交接说明

日期：2026-06-18

## 1. 目的

这份文档用于把本轮 SWE Pro full-repo retrieval-only 问题的前因后果整理清楚，方便后续由代码 RAG 侧做专项修复。

重点结论先放前面：

1. 最初的“NL 查询没有切到 NL/Issue 权重”问题是真实存在的，已经修复。
2. 修复路由后，SWE Pro 命中率依然偏低，说明主问题已经转移到 issue 查询的召回和排序层，不再是动态权重切换本身。
3. 当前冻结权重本身不建议改，下一步应该继续修 issue 查询的候选生成、测试文件抑制、config/doc 多目标检索。

---

## 2. 冻结基线架构

本轮排障默认以这套冻结策略为准，不改融合比例，只修路由和排序异常：

### 2.1 Query Profile 与权重

| 场景 | Graph | Vector | BM25 |
|---|---:|---:|---:|
| Code / Symbol / Function memory | 0.90 | 0.05 | 0.05 |
| NL / Issue / Bug report | 0.30 | 0.50 | 0.20 |

### 2.2 其他冻结项

- `rrf_k = 60`
- CallGraph 开启
- FTS5 / BM25 开启
- Jina 768d 本地向量链路保留
- INT8 向量保留
- P1 ReRanker 保留

### 2.3 本次实际运行的代码源

SWE Pro 验证脚本实际加载的是独立包：

- [F:\codegraph-rag\src\codegraph_rag\search.py](/F:/codegraph-rag/src/codegraph_rag/search.py:1)

不是 harness 镜像副本：

- [F:\新建文件夹\deer-flow-main\backend\packages\harness\deerflow\tools\codegraph\search.py](/F:/新建文件夹/deer-flow-main/backend/packages/harness/deerflow/tools/codegraph/search.py:1)

这点非常关键。前面有一轮“看起来修了但结果没变”，根因就是补丁先打到了 harness 副本，而不是实际运行包。

---

## 3. 问题时间线

## 3.1 阶段 A：初始异常确认

报告：

- [codegraph_retrieval_validation_swepro4_20260618_053359.md](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_swepro4_20260618_053359.md)
- [codegraph_retrieval_validation_swepro4_20260618_053359.jsonl](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_swepro4_20260618_053359.jsonl)

结果：

- `P@1 = 25%`
- `Hit@3 = 25%`
- `Hit@5 = 25%`
- `Hit@10 = 25%`
- `MRR = 0.25`
- `target_miss_at_10 = 3`

当时最关键的异常证据是：

- `query_type = nl`
- 但 `query_profile = function_memory`
- 最终权重仍然是 `graph=0.9 / vector=0.05 / bm25=0.05`

这和冻结策略明显冲突，因为 SWE issue 查询应该走 `issue` profile，也就是 `0.3 / 0.5 / 0.2`。

---

## 3.2 阶段 B：仅修路由后复测

报告：

- [codegraph_retrieval_validation_20260618_054804.md](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_054804.md)
- [codegraph_retrieval_validation_20260618_054804.jsonl](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_054804.jsonl)

结果：

- `P@1 = 0%`
- `Hit@3 = 25%`
- `Hit@5 = 25%`
- `Hit@10 = 25%`
- `MRR = 0.125`
- `target_miss_at_10 = 3`

这一步说明两件事：

1. 路由确实修到了：
   - 4 条样本都已经切到 `query_profile = issue`
   - 权重也变成了 `0.3 / 0.5 / 0.2`
2. 命中率没有因此抬起来：
   - 说明后续瓶颈不在“有没有切到 NL/Issue 权重”
   - 而在 issue 查询的召回和排序逻辑

典型异常：

- NodeBB：`graph = 0`，图检索几乎没有有效候选
- qutebrowser / ansible：测试文件和测试符号被排到实现文件前面
- qutebrowser changelog case：top 结果基本被测试和无关函数占住

---

## 3.3 阶段 C：补 issue-query 检索逻辑后复测

报告：

- [codegraph_retrieval_validation_20260618_061404.md](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_061404.md)
- [codegraph_retrieval_validation_20260618_061644.md](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_061644.md)
- [codegraph_retrieval_validation_20260618_061644.jsonl](/F:/codex-cache/benchmarks/codegraph_retrieval_validation_20260618_061644.jsonl)

最终有效结果：

- `P@1 = 25%`
- `Hit@3 = 25%`
- `Hit@5 = 50%`
- `Hit@10 = 50%`
- `MRR = 0.3125`
- `target_miss_at_10 = 2`
- 平均热检索延迟约 `260ms`

相对阶段 B 的变化：

- `Hit@5: 25% -> 50%`
- `Hit@10: 25% -> 50%`
- `MRR: 0.125 -> 0.3125`

说明 issue 查询侧的补丁方向是有效的，但还没有修到“稳定命中”。

---

## 4. 已确认根因

## 4.1 根因一：Query profile 优先级错误

问题位置：

- [search.py `_detect_query_profile`](/F:/codegraph-rag/src/codegraph_rag/search.py:849)

旧行为：

- `function_memory` 的判断优先于 `issue`
- SWE issue 文本里常出现 `Description:`、`function`、`Failing tests:` 之类词
- 导致真实 issue 被误归类成 `function_memory`

后果：

- 虽然 `query_type = nl`
- 但 `query_profile` 错了
- 最终沿用 graph-first 权重

这就是最开始 “为什么 query_type=nl 却用了 0.9/0.05/0.05” 的直接原因。

---

## 4.2 根因二：权重选择没有复用最终 profile

问题位置：

- [search.py `_get_dynamic_weights`](/F:/codegraph-rag/src/codegraph_rag/search.py:940)

旧行为：

- `search()` 已经得到 `qtype` 和 `profile`
- 但 `_get_dynamic_weights(query)` 又重新基于原始 query 做了一次判断

后果：

- 路由和权重选择可能漂移
- explain trace、fusion、reranker 的语义不一致

这也是为什么早期看 trace 时会出现“上层看起来是 NL，但底层仍然拿 graph-heavy 权重”的第二层原因。

---

## 4.3 根因三：Issue 原文太脏，图检索吃到噪声

问题位置：

- [search.py `_normalize_query_text`](/F:/codegraph-rag/src/codegraph_rag/search.py:416)
- [search.py `_split_issue_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:434)
- [search.py `_build_issue_focus_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:460)
- [search.py `_build_issue_test_hint_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:515)

旧行为：

- 原始 query 直接拿整段 issue 文本去检索
- 里面混有：
  - 转义换行
  - 标题 / labels / steps to reproduce
  - 大段 failing tests
  - 文件名 / testcase 名 / 参数值

后果：

- graph 查询提取出很多脏符号
- graph channel 不是“找实现文件”，而是在“找测试符号或噪声词”
- 典型表现就是 NodeBB 早期 `graph_count = 0`，或者 graph 结果高度偏 test 名称

---

## 4.4 根因四：Failing tests 对测试文件有过强牵引

问题位置：

- [search.py `search()` issue 图检索分流逻辑](/F:/codegraph-rag/src/codegraph_rag/search.py:48)
- [reranker.py issue-mode penalty](/F:/codegraph-rag/src/codegraph_rag/reranker.py:733)

SWE Pro 的 query 往往会带：

- `Failing tests:`
- 测试函数名
- 测试文件路径
- traceback 样式内容

如果直接把这部分当主查询：

- BM25 会强命中测试文件
- 图检索会围着测试符号转
- reranker 在没有惩罚时也会把 test chunk 排得很靠前

这就是 qutebrowser 和 ansible 初期“结果看起来像 test retriever，不像 implementation locator”的根因。

---

## 4.5 根因五：Context prior 不足以拯救坏候选池

问题位置：

- [search.py `_apply_context_prior`](/F:/codegraph-rag/src/codegraph_rag/search.py:402)

现状：

- SWE Pro full-repo retrieval-only 场景里，`active_file = null`、`language = null`
- context 里常常只有 repo/workspace 级别信息

后果：

- context prior 只能做软加分
- 它无法把一个本来就没进 Top50 的目标文件拉回来
- 更无法解决多目标 issue、doc/config/schema 类型目标缺失的问题

所以这里不是主 bug，但它也解释了为什么“context 看着开了，结果还是救不回来”。

---

## 5. 已落地修复

## 5.1 路由修复

已在 runtime 包中完成：

- `issue` 判断前移，优先级高于 `function_memory`
- `_get_dynamic_weights()` 复用已解析好的 `profile`
- 扩展强 issue marker：
  - `current behavior`
  - `failing tests:`
  - `failing test:`
  - `test failure`
  - `stack trace`

关键位置：

- [search.py `_detect_query_profile`](/F:/codegraph-rag/src/codegraph_rag/search.py:849)
- [search.py `_get_dynamic_weights`](/F:/codegraph-rag/src/codegraph_rag/search.py:940)

结果：

- 4 条 SWE 样本都能稳定切到 `query_profile = issue`
- 权重稳定为 `0.3 / 0.5 / 0.2`

---

## 5.2 Issue 查询清洗与双通道图提示

已在 runtime 包中完成：

1. 先做 benchmark 风格文本归一化
2. 把 issue body 与 failing tests 拆开
3. 主图检索只吃清洗后的 issue body
4. failing tests 只提取少量 implementation-oriented hints，作为次级 graph 补充
5. explain trace 里显式输出：
   - `graph_query`
   - `issue_test_hints`

关键位置：

- [search.py `_normalize_query_text`](/F:/codegraph-rag/src/codegraph_rag/search.py:416)
- [search.py `_split_issue_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:434)
- [search.py `_build_issue_focus_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:460)
- [search.py `_build_issue_test_hint_query`](/F:/codegraph-rag/src/codegraph_rag/search.py:515)
- [search.py `_build_explain_trace`](/F:/codegraph-rag/src/codegraph_rag/search.py:1437)

---

## 5.3 Issue 模式下对测试候选做轻量抑制

已在 reranker 中加了小幅 test penalty：

- 如果候选路径 / symbol / tag 明显像测试文件
- 在 `issue_mode` 下减去 `0.035`

关键位置：

- [reranker.py issue penalty](/F:/codegraph-rag/src/codegraph_rag/reranker.py:733)

这一步不是“修所有问题”的银弹，但它已经帮助 qutebrowser moved-test case 从“test 文件压前”转成“实现文件进入前排”。

---

## 6. 当前仍未解决的问题

## 6.1 NodeBB：多目标 issue，单一 query 很难覆盖

当前命中情况：

- `target_miss_at_10`

目标文件横跨：

- controller
- database
- socket
- user
- tpl
- i18n json
- openapi yaml

这是典型的多目标 issue，不是单函数定位问题。

当前 top 结果更多落在：

- `src/install.js`
- `src/upgrades/...`
- `src/cli/upgrade.js`
- `public/src/utils.common.js`

说明现在的 issue 检索仍然偏“语义相近的代码块”，而不是“按照问题面分解出多个文件簇”。

---

## 6.2 qutebrowser changelog/config case：doc/config 类型目标覆盖不足

当前命中情况：

- `target_miss_at_10`

期望目标里含有：

- `qutebrowser/app.py`
- `qutebrowser/config/configfiles.py`
- `qutebrowser/config/configdata.yml`
- `doc/changelog.asciidoc`
- `doc/help/settings.asciidoc`

但当前前排更多是：

- `qutebrowser/misc/sql.py`
- `qutebrowser/utils/log.py`
- `qutebrowser/config/configutils.py`

这说明：

1. 当前函数级索引更偏代码实现
2. doc / config / yaml / changelog 目标在 issue 模式下没有被足够提升
3. 测试名和 version/config 词还会把结果带偏

---

## 6.3 Harness 镜像与 runtime 包仍有同步差

当前确认状态：

- `F:\codegraph-rag` 已有 route fix + retrieval fix
- harness 副本里已能看到 route fix 测试
- 但 retrieval-focused 补丁没有完全同步进去

从现有 harness `search.py` 看，仍然是旧检索流程：

- 没有 `_split_issue_query`
- 没有 `_build_issue_focus_query`
- 没有 `_build_issue_test_hint_query`

因此后续云端或主工程同步时，要以 `F:\codegraph-rag` 为源头，不要反向覆盖。

---

## 7. 对最初 5 个排查问题的结论

### 7.1 为什么 `query_type=nl` 时仍用了 `0.9 / 0.05 / 0.05`？

因为当时真正决定权重的是 `query_profile`，而不是 `query_type`。

旧代码里：

1. `_detect_query_profile()` 会把 issue 误判成 `function_memory`
2. `_get_dynamic_weights()` 又重新基于原始 query 做判断

所以最终落到了 graph-first 家族。

这个问题已经修复。

### 7.2 SWE Pro issue 查询应该如何稳定走 NL/Issue 权重？

当前可行且已落地的方法就是：

1. `issue` 优先级高于 `function_memory`
2. `search()` 里一旦解析出 `profile=issue`，后续权重选择必须复用该结果
3. `Failing tests / traceback / current behavior` 这些标记都直接视为强 issue signal

### 7.3 是否需要把 issue / failing tests / traceback 单独识别为 issue 模式？

需要，而且已经这样做了。

更准确地说，不是只“识别成 issue”，还要进一步：

1. 把 issue body 作为主检索语义
2. 把 failing tests 作为辅助 hint
3. 不要让 failing tests 直接主导整个 candidate pool

### 7.4 是否是 reranker / context_prior 把目标实现文件压下去了？

部分是，但不是唯一原因。

更完整的说法是：

1. 前期主问题是候选池本身就坏了
2. reranker 之前没有抑制测试文件，确实会把 test chunk 顶上来
3. context prior 在 SWE Pro 这种弱上下文场景里能力有限，只能微调，不能救坏召回

### 7.5 最小修复建议是什么？

在不改冻结权重的前提下，最小且有效的下一步是：

1. 保持现有 route fix
2. 保持 issue body / failing tests 拆分
3. 继续增强 issue-mode 的 candidate prior
4. 优先解决 doc/config/multi-target 两类剩余 miss

---

## 8. 建议的下一步修复 backlog

按优先级建议这样拆：

### P0：Issue 双路候选池固化

目标：让 body 语义和 failing-test hints 各自召回，不互相污染。

建议做法：

1. `problem statement lane`
   - 只吃清洗后的 issue body
   - 负责实现文件、配置文件、文档文件的主召回
2. `failing test lane`
   - 只吃提取后的 test hints
   - 负责少量 graph 补召和 symbol 靠拢
3. lane 内先各自排序，再做受控融合
4. 为每个 lane 设上限，避免测试 lane 把主 lane 淹掉

### P0：Issue 模式候选类型先验

目标：让实现 / config / docs 在 issue 模式下比 tests 更容易进前排。

建议做法：

1. issue 模式对下列路径给轻量 boost：
   - `src/`
   - `lib/`
   - `qutebrowser/config/`
   - `doc/`
   - `public/language/`
   - `openapi/`
2. 对显著测试路径继续保留 penalty
3. 对 `yml/yaml/tpl/asciidoc/json` 这类非函数型目标增加类型感知加分

### P1：Multi-target issue 分解

目标：修 NodeBB 这种“一个 issue 关联很多文件簇”的问题。

建议做法：

1. 从 issue body 中抽取多个子意图：
   - ACP / admin users
   - email validation
   - confirmation keys
   - fallback lookup
   - UI / i18n / schema
2. 每个子意图单独召回局部 TopN
3. 再对文件级去重聚合

这一步会比继续微调当前单查询 rerank 更有效。

### P1：Config / Doc / Changelog 专项 booster

目标：修 qutebrowser changelog case。

建议做法：

1. 如果 issue 文本出现：
   - `changelog`
   - `version`
   - `settings`
   - `config`
   - `upgrade`
2. 则对以下目标类做额外 boost：
   - `config*.py`
   - `config*.yml`
   - `doc/*.asciidoc`
   - `app.py`

这类 case 的问题不是“没有语义”，而是“正确目标类型没有被纳入优先面”。

---

## 9. 建议保留不动的部分

以下内容本轮不建议再碰：

1. 冻结权重
   - Code/Symbol: `0.9 / 0.05 / 0.05`
   - NL/Issue: `0.3 / 0.5 / 0.2`
2. `rrf_k = 60`
3. 三模结构本身
4. Jina 向量链路
5. 现有 reranker 主体权重

原因很简单：当前问题已经明确不是“融合比例不对”，继续改权重只会把真正的检索缺陷掩盖掉。

---

## 10. 需要同步的文件

本轮交接建议重点看这些文件：

### Runtime 包

- [search.py](/F:/codegraph-rag/src/codegraph_rag/search.py:1)
- [reranker.py](/F:/codegraph-rag/src/codegraph_rag/reranker.py:1)
- [test_search_routing.py](/F:/codegraph-rag/tests/test_search_routing.py:1)

### Harness 镜像

- [search.py](/F:/新建文件夹/deer-flow-main/backend/packages/harness/deerflow/tools/codegraph/search.py:1)
- [test_codegraph_search_routing.py](/F:/新建文件夹/deer-flow-main/backend/tests/test_codegraph_search_routing.py:1)

---

## 11. 复现与注意事项

1. 看 SWE Pro 当前有效结论，请以 `061644` 这版报告为准。
2. 当前机器这次没有在系统 Python 上重跑单测，因为缺少 `networkx` 依赖；这不影响已有 artifact 分析结论，但如果要继续开发，请切到项目依赖完整的环境里复测。
3. 同步代码时优先以 `F:\codegraph-rag` 为主，不要被 harness 副本的旧逻辑反向覆盖。

---

## 12. 一句话结论

这次 SWE Pro 问题已经确认分成两层：

1. 路由层 bug：已修复
2. 检索层 issue-mode 候选质量不足：仍需专项修

后续专项修复应聚焦：

- issue body / failing tests 双路召回
- 实现文件与 doc/config 文件优先面
- 多目标 issue 分解

而不是继续改冻结权重。
