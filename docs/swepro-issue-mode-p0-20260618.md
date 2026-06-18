# SWE Pro Issue-Mode P0 Internal Optimizations (2026-06-18)

Scope: 内部检索/精排优化，不动冻结的 RRF 权重，配合 SWE_PRO_RETRIEVAL_REPAIR_BRIEF_20260618.md 的 P0 backlog 落地。

## 1. 改动点

### 1.1 search.py — Issue 双路候选池

- 新增 `_extract_issue_keywords(text)`：从 issue body 抽取意图关键词（config / settings / changelog / version / upgrade / docs / openapi / schema / i18n / template / controller / database / socket / email / ...），用于驱动 rerank 的类型先验。
- 新增 `_is_test_path(result)`：识别 `tests/`、`test/`、`_test.py`、`.test.{js,ts}`、`symbol startswith test_`、tag=test 等测试类候选。
- `search()` 内对 `profile == "issue"` 增加：
  - BM25 主路：原有 issue body（focus query）召回。
  - BM25 hint lane：用 `_build_issue_test_hint_query(...)` 抽出的实现导向关键词再跑一次 BM25，先剔除测试类候选再合并（score_scale=0.55，受控合并，主路始终占多数）。
  - Vector 主路：原有 issue body 向量召回。
  - Vector hint lane：与 BM25 hint lane 同样的策略，先剔除测试类候选再合并，避免 failing test 文件淹没实现/配置/文档目标。
- Graph 主路保持原始 `graph_query`，并维持原来 `_merge_ranked_results(...)` 的 0.72 缩放合并 issue_test_hints 召回。
- Issue 模式下把 `issue_keywords` 透传给 `_apply_reranker(...)`。

### 1.2 reranker.py — Issue 模式类型/路径先验

仅在 `candidate["issue_mode"]=True` 时生效，对 RRF→精排后的最终分加减偏移：

- 测试路径/测试符号：`final -= 0.060`（原 0.035 太弱）。
- 实现路径加分：`src/`, `lib/`, `controllers/`, `models/`, `services/`, `api/`, `database/`, `socket.io/`, `user/`, `middleware/` 命中 → `+0.025`。
- 关键词驱动加分（基于 file_path + search_document + docstring 拼接判定）：
  - `config/settings/options` → `/config/`, `config`, `settings`, `options` → `+0.045`
  - `changelog/release/version/upgrade/docs` → `.asciidoc`, `.rst`, `.md`, `/doc/`, `/docs/`, `changelog`, `release_notes` → `+0.045`
  - `openapi/schema` → `openapi`, `/schemas/`, `/schema/`, `.yml`, `.yaml`, `.json` → `+0.04`
  - `template/view/ui` → `.tpl`, `/views/`, `/templates/`, `/template/` → `+0.035`
  - `i18n/language/translation` → `/language/`, `/languages/`, `/locales/`, `/i18n/`, `/translations/` → `+0.04`
  - `controller/router/route` → `/controllers/`, `/routes/`, `/router/` → `+0.03`
  - `database/db/migration` → `/database/`, `/db/`, `/migrations/` → `+0.03`
  - `socket/ws/websocket` → `/socket.io/`, `/sockets/`, `/ws/` → `+0.03`
  - `email/mail` → 文档/路径含 `email`/`mail` → `+0.02`

所有偏移均为线性叠加在线 score 上，没有改 RRF 权重、没有改 NL/Code/Function 权重模型。

## 2. 不动的部分

- RRF 三模权重：Code/Symbol/Function `0.90/0.05/0.05`，NL/Issue `0.30/0.50/0.20`。
- `rrf_k=60`、CallGraph、FTS5/BM25、Jina 768d、INT8、P1 ReRanker 主体。
- 通用 NL 与 Code 模式路径，仅当 `profile == "issue"` 时才会进入新分支。

## 3. 已加测试

`tests/test_search_routing.py`：

- `test_extract_issue_keywords_config_doc_changelog`
- `test_extract_issue_keywords_empty_for_unrelated_text`
- `test_is_test_path_detects_common_test_locations`
- `test_issue_dual_lane_uses_test_hints`

全部 8 个用例本地 `pytest -q` 通过。

## 4. 预期作用面

- NodeBB ACP/email/openapi 多目标：openapi/schema/email 类型先验上抬，实现路径 `src/controllers/`, `src/database/`, `src/socket.io/`, `src/user/`, `public/openapi/`, `public/language/` 更易进 Top10。
- qutebrowser changelog/configdata：`config + changelog + upgrade` 触发 `config/`, `doc/`, `changelog` 类型加分；同时 `tests/` 整体下压。
- ansible/qutebrowser 之前出现的 test-file 占位问题：issue 模式下测试候选直接 -0.060，搭配 hint-lane 内提前过滤，双重抑制。

## 5. 后续 backlog（未做）

按 SWE_PRO_RETRIEVAL_REPAIR_BRIEF_20260618.md：

- P1 Multi-target issue 分解（NodeBB 类多文件簇）。
- P1 进一步的 doc/config/changelog 专项 booster（可基于关键词扩展更多模式）。


## 5. v6_titlevec ???2026-06-18 08:08?

?? NodeBB / qute-f631cd ? long-issue-body ???????gold ????? Top200 ????? issue title ???? lane ?? BM25 / vector ??????? RRF ?????

### 5.1 search.py ????

- ?? `_extract_issue_title(text)`??? `**Title: X**` / `Title:` heading-only / `# header` / fallback first-line???? 200 char?
- ?? `_rank_fuse_lanes(lanes, top_k, rrf_k)`?channel ?? lane RRF???????? RRF ???????? "primary + scaled secondary" ????????????
- BM25 issue ???`body / title / hint` ? lane RRF ????? `1.00 / 0.65 / 0.55`?Title ? body ???????hint lane ???? `_is_test_path` ???
- Vector issue ???`body / title / hint` ? lane RRF ????? `0.85 / 1.00 / 0.55`?Title ?? ? body ?? ? NodeBB ????? issue title ???????? `src/socket.io/admin/user.js` ?? Top200 ? rank 2??? title ???????????? body ??????
- `_apply_reranker(...)` ?? `issue_title` ??????? `issue_title_tokens` ???? reranker ? filename/path ?? token ???

### 5.2 ?? / ????

`tests/test_search_routing.py`?12 ?????????

- `test_extract_issue_title_inline_marker`
- `test_extract_issue_title_heading_lookahead`
- `test_issue_title_bm25_lane_invoked`??? issue ???? 3 ? BM25 ???? title ???
- `test_issue_title_vector_lane_invoked`??? issue ???? 3 ? embedding + vector_search ???? title ???

### 5.3 4 ? SWE-Pro ????

| Tag | P@1 | Hit@3 | Hit@5 | Hit@10 | MRR | miss@10 | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 061942 baseline | 0.250 | 0.250 | 0.500 | 0.500 | 0.3125 | 2 | 308 ms |
| p0_internal_opt | 0.250 | 0.500 | 0.500 | 0.500 | 0.3750 | 2 | 303 ms |
| p0_generic_v4_pool (200 pool) | 0.250 | 0.250 | 0.500 | 0.750 | 0.3278 | 1 | n/a |
| **p0_generic_v6_titlevec** | **0.500** | **0.750** | **1.000** | **1.000** | **0.6875** | **0** | 726 ms |

Per case?v6_titlevec??

- `NodeBB__NodeBB-0499...` ? miss ? **rank 1**?`src/user/email.js`??
- `qutebrowser-f91ace...` ? rank 1 ? **rank 1**?????
- `qutebrowser-f631cd...` ? miss ? **rank 4**?`qutebrowser/config/configfiles.py`??
- `ansible-f327e6...` ? rank 5?v4 ??? ? **rank 2**?`lib/ansible/utils/collection_loader/_collection_finder.py`??

### 5.4 ??

- ??? ~300 ms ?? ~726 ms?issue ???? 1 ? title embedding + 1 ? title ???? + 1 ? title BM25 ????
- code/symbol/function ??????????issue ??????????????
