# Legal Application Benchmark

Hướng dẫn đầy đủ từng lệnh với checkpoint local Qwen3-4B, môi trường Python hiện có và concurrency 20: [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md).

Đánh giá năng lực áp dụng pháp luật của LLM trên bản án Việt Nam. Mỗi bản án được gửi trong **một request generator**, gồm toàn bộ facts_clean, issues và luật theo thứ tự gốc. Bài làm là văn bản tự do, giải quyết và đưa ra phán quyết **riêng cho từng issue**. Không yêu cầu phán quyết cuối cùng cho toàn vụ án.

Judge đọc đầy đủ nguồn tham chiếu, suy ra kết quả từng issue trực tiếp từ application/conclusion, gán nhãn và giải thích. Python kiểm tra schema, mapping và tính điểm. Không xây Legal Issue Tree hoặc một pipeline tạo gold riêng.

## Cài đặt trên B200

Client cần Python **>=3.10**, không cần GPU hay PyTorch. Dùng môi trường `.venv` hiện có của bạn, đặt ở bất kỳ thư mục nào. Script gọi `python` trên PATH sau khi activate môi trường; không kiểm tra phiên bản thư viện hoặc yêu cầu `.venv` trong project. Server vLLM được triển khai riêng.

```bash
# Activate môi trường .venv hiện có của bạn trước khi chạy các lệnh này.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/minhpn19/LegalApplication}"
cd "$PROJECT_ROOT"
```

Nếu môi trường đã có đầy đủ dependencies, chạy trực tiếp entry point. Khi cần cài vào một môi trường mới, dùng `python -m pip install -r requirements.txt`; `requirements-dev.txt` bao gồm dependencies runtime và pytest. Script chạy trực tiếp package từ `src/`; nếu muốn dùng thêm lệnh `legal-application` sau khi cài package, có thể cài `python -m pip install -e .`. Dependencies cũng được khai báo trong `pyproject.toml`. vLLM được cài riêng trong môi trường host model.

Entry point chính: `bash "$PROJECT_ROOT/scripts/run_application.sh"`. Script không tự cài hoặc khởi động model. Paths tương đối được resolve từ PROJECT_ROOT, kể cả khi gọi từ thư mục khác. Path tuyệt đối do người dùng export được giữ nguyên.

## Dữ liệu và bước kiểm tra trước khi chạy

Mỗi dòng JSONL phải là một object với:

| Trường | Contract |
|---|---|
| `doc_name` | Chuỗi định danh không rỗng, duy nhất trong file |
| `benchmark_seed.facts_clean` | Chuỗi không rỗng, được đưa vào generator |
| `benchmark_seed.facts`, `.application`, `.conclusion` | Bắt buộc tồn tại và là chuỗi; được giữ nguyên cho Judge. Chuỗi tham chiếu rỗng có thể dẫn đến needs_review |
| `issues` | Danh sách chuỗi không rỗng, giữ nguyên nội dung và thứ tự |
| `laws` | Danh sách luật không rỗng; mỗi luật có `law_str` |
| `laws[].provisions[]` | Nếu có provisions: bắt buộc `citation_str` và `law_content.context_text` không rỗng cho từng provision |
| `laws[].law_content` | Khi không có provisions: dùng `context_text` nếu khác null; nếu thiếu/null thì dùng `exact_text`. Nội dung được chọn phải là chuỗi không rỗng |

Fallback chỉ áp dụng ở **cấp luật khi không có provisions**; không tự thay context_text của provision bằng exact_text. Nguồn fallback được ghi trong prompt, `metadata.law_sources` của generation và run.log. Không chọn lọc luật theo mức liên quan, không cắt ngắn hoặc chuẩn hóa lại nội dung. Generator dùng projection allowlist, không nhận application, conclusion, markdown, explain hoặc metadata của luật chứa nguồn đáp án. Judge nhận toàn bộ benchmark_seed, issues, luật gốc và nguyên văn message.content của generator.

Trước mọi request, pipeline quét **toàn bộ file**, kể cả phần ngoài LIMIT. Dòng không parse được strict JSON/UTF-8 (kể cả dòng trắng, key JSON trùng và NaN/Infinity) được ghi đầy đủ vào `OUTPUT_DIR/RUN_ID/removed_input_lines.jsonl`, sau đó **xóa vĩnh viễn khỏi chính INPUT_JSONL**. Pipeline tiếp tục với các dòng hợp lệ phía sau; những lần chạy sau đọc file đã làm sạch nên không gặp lại các dòng đã xóa. LIMIT được áp dụng lên các samples còn lại, không tính các dòng bị xóa vào quota.

Log có đường dẫn input, thời gian, số dòng/index gốc, loại lỗi/lý do/vị trí lỗi, nguyên văn dòng, bản base64 của toàn bộ bytes và checksum. `input_cleanup.json` lưu hash trước/sau, số dòng đã xóa và trạng thái transaction. Journal được ghi atomic và fsync **trước khi** thay input bằng file sạch; nếu ghi log thất bại, input được giữ nguyên và chưa gọi model. File sạch giữ nguyên bytes và thứ tự của các dòng còn lại. Lock riêng ở `<INPUT_JSONL>.cleanup.lock` ngăn hai process đồng thời làm sạch cùng file.

Sau cleanup mới validate schema và đánh số samples. `input_index` là số dòng vật lý **bắt đầu từ 0 của file đã làm sạch**; `input_line` bắt đầu từ 1. Vì thế index ổn định khi resume hoặc chạy lại file sạch; số dòng trước khi xóa nằm trong removal log. JSON hợp lệ nhưng sai schema, thiếu trường bắt buộc, doc_name trùng hoặc thiếu nguồn luật vẫn được giữ nguyên và làm preflight thất bại. Nếu không còn sample nào sau cleanup, pipeline lưu log và báo lỗi input rỗng.

Hành vi làm sạch áp dụng cho cả `full`, `judge-only` và `validate-only`. `validate-only` không gọi model nhưng **có sửa file input** khi phát hiện JSON lỗi. Ví dụ:

```bash
MODE=validate-only RUN_ID=schema-check \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Tệp mẫu ban đầu có 6 records hợp lệ và một chuỗi JSON chưa đóng ở dòng 7. Với hành vi mới, lệnh trên lưu dòng 7 vào removal log, xóa dòng đó khỏi input và validate 6 records còn lại. Xem `counts.input_removed` trong summary để biết số dòng bị xóa. Để smoke test riêng 1–5 bản án, có thể tạo một tệp con chứa 5 dòng đầu của input đã làm sạch:

```bash
python - <<'PY'
import os
from itertools import islice
from pathlib import Path
root = Path(os.environ['PROJECT_ROOT'])
with (root / 'input/judgment_with_issues_sample.jsonl').open('rb') as source:
    lines = list(islice(source, 5))
with (root / 'input/smoke_5.jsonl').open('xb') as target:
    target.writelines(lines)
PY

MODE=validate-only INPUT_JSONL=input/smoke_5.jsonl RUN_ID=smoke-schema \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

## Endpoint và xhigh

Generator dùng OpenAI-compatible `POST <GENERATOR_BASE_URL>/chat/completions`, không ép JSON. Phần đầu `scripts/run_application.sh` có hai exports để đổi model được đánh giá. Script hiện điền sẵn ví dụ Qwen3-4B; bạn có thể sửa giá trị sau `:-` trực tiếp trong file hoặc export từ terminal để ghi đè:

```bash
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-http://127.0.0.1:8001/v1}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-4B}"
```

`GENERATOR_BASE_URL` là địa chỉ API của máy host model, có `/v1` ở cuối. Nếu pipeline và server cùng máy B200, dùng `127.0.0.1`; nếu server ở máy khác, thay bằng IP/hostname máy đó. `GENERATOR_MODEL` là tên model endpoint phục vụ, phải khớp `--served-model-name`, không phải link tải checkpoint. Biến đã export ở terminal có ưu tiên cao hơn giá trị mặc định trong script.

Ví dụ mở một terminal riêng trên B200 để host Qwen3-4B:

```bash
vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --host 0.0.0.0 --port 8001 \
  --reasoning-parser qwen3
```

Parser `qwen3` tách phần reasoning khỏi final content theo [tài liệu triển khai Qwen với vLLM](https://qwen.readthedocs.io/en/stable/deployment/vllm.html). `GENERATOR_THINKING_MODE=none` trong client chỉ có nghĩa không gửi tham số reasoning riêng; không tắt thinking mặc định của Qwen. Script không tự tải, khởi động hoặc host checkpoint. Nếu dùng checkpoint khác đã có, có thể triển khai bằng ví dụ khung sau trong môi trường server riêng:

```bash
: "${MODEL_PATH:?Đặt đường dẫn checkpoint đã được cung cấp}"
: "${GENERATOR_MODEL:?Đặt tên model được serve}"
vllm serve "$MODEL_PATH" \
  --served-model-name "$GENERATOR_MODEL" \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size "${TP_SIZE:-1}"
```

Cấu hình tensor parallel, context length, tokenizer và reasoning parser phải phù hợp checkpoint/hạ tầng. Endpoint cần đủ context cho **toàn bộ** văn bản và output budget; khi bị từ chối do context quá dài, pipeline ghi lỗi sample, không cắt dữ liệu và không giảm budget ngầm.

Judge mặc định cố định là `DeepSeek-V4.1-Flash-1017` tại `http://172.29.62.173:8000/v1`, effort xhigh. Cách gửi được đối chiếu trực tiếp với `LegalDataset/src/legal_issue_pipeline/llm/adapters.py`, `DeepSeekVLLMAdapter` và `tests/unit/test_vllm_effort.py` trong repository lân cận:

```json
{"chat_template_kwargs":{"thinking":true,"reasoning_effort":75}}
```

Không gửi `reasoning_effort="xhigh"` ở top level và không dùng cách map của hosted DeepSeek API. Budget 75 là tham số effort, độc lập với `max_tokens`. Mặc định generator không gửi tham số reasoning riêng và dùng thiết lập của server/template. Nếu generator cũng là DeepSeek vLLM tương ứng, đặt `GENERATOR_THINKING_MODE=deepseek_vllm_xhigh` để gửi đúng body trên.

Judge mặc định dùng `response_format.type=json_schema`, `strict=true`. Nếu endpoint chỉ hỗ trợ JSON object, có thể **cấu hình rõ trước run mới** `JUDGE_RESPONSE_FORMAT=json_object`; pipeline vẫn validate cùng schema ở phía Python. Không tự fallback sau HTTP 400, không tắt xhigh ngầm. `message.reasoning_content` không được đọc, lưu hoặc chuyển vai trò; raw response trong artifacts là nguyên văn `message.content`, không strip whitespace. Endpoint reasoning cần tách final content đúng cách.

## Chạy thử trên B200 và chạy full

Chỉ thực hiện các lệnh có inference trên máy runtime đã được chỉ định. Trong quá trình phát triển chỉ chạy mock tests và validate-only.

```bash
export GENERATOR_BASE_URL='http://127.0.0.1:8001/v1'
export GENERATOR_MODEL='TEN_MODEL_THUC_TE_DA_SERVE'
export INPUT_JSONL='input/smoke_5.jsonl'

# 1 sample
MODE=full LIMIT=1 RUN_ID=smoke-1 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"

# 5 samples; vẫn xử lý đủ dù ít hơn BATCH_SIZE=20
MODE=full LIMIT=5 BATCH_SIZE=20 RUN_ID=smoke-5 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"

# Toàn bộ dataset ĐÃ SỬA/VALIDATE; không có cap mặc định 1000
MODE=full INPUT_JSONL=input/full_dataset.jsonl LIMIT=all \
  BATCH_SIZE=20 GENERATOR_CONCURRENCY=20 JUDGE_CONCURRENCY=20 \
  RUN_ID=full-run bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Với endpoint hỗ trợ output dài hơn, export `MAX_OUTPUT_TOKENS=65536`. Giá trị này được gửi nguyên vẹn dưới tên `max_tokens` cho cả hai vai trò. Response `finish_reason=length`, filtered hoặc final content rỗng không được nhận là generation thành công; Judge có thể regenerate trong structured-output retry budget. Pipeline không tái sinh generation chỉ vì bài làm bỏ sót issue hay sai format, vì đó là hành vi cần được Judge đánh giá.

## Judge-only và resume

Judge-only dùng JSONL do pipeline đã lưu; không gọi generator. Mọi record được dùng phải khớp doc_name, input_index, sample checksum và response checksum. Nếu file có nhiều records hơn LIMIT, chỉ lấy records của các sample được chọn, không dựa vào vị trí dòng trong generation file. Model gốc được lấy từ từng generation record.

```bash
MODE=judge-only INPUT_JSONL=input/full_dataset.jsonl \
  GENERATION_JSONL=outputs/full-run/generations.jsonl \
  LIMIT=all RUN_ID=judge-rerun \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Một selected sample không có generation được báo failed riêng trong errors.jsonl; các sample còn lại vẫn được chấm. Generation source hỏng JSON, sai mapping hoặc trùng identity làm preflight dừng trước API. Judge-only tạo bản sao nguyên record generation được dùng trong run directory mới.

Mỗi RUN_ID có thư mục riêng. Nếu thư mục đã tồn tại, mặc định dừng, không ghi đè. Resume bằng đúng cấu hình/input/code của run cũ và thêm `RESUME=1`:

```bash
# Các export và tham số phải giống run smoke-5 ở trên.
MODE=full LIMIT=5 BATCH_SIZE=20 RUN_ID=smoke-5 RESUME=1 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Snapshot khóa cả LIMIT, batch/concurrency, endpoints, model, output budget, hash toàn bộ **input đã làm sạch**, generation source và code/prompt/schema. Snapshot này được lưu trước khi commit cleanup và trước mọi request. Resume kiểm tra hash sau khi loại các dòng JSON lỗi: nếu nội dung/thứ tự các dòng hợp lệ vẫn như cũ thì tiếp tục an toàn; nếu valid records, cấu hình hoặc code khác thì từ chối resume **trước khi sửa input** và yêu cầu RUN_ID mới. API keys có thể thay mà không ảnh hưởng resume và không được lưu trong snapshot. Kết quả Judge `completed` và `needs_review` đều là terminal: resume bỏ qua cả hai. Muốn chấm lại review, dùng judge-only với RUN_ID mới.

Generation được append và fsync ngay sau response thành công, trước khi chuyển sang Judge. Nếu Judge lỗi, resume chỉ gọi lại Judge. Mỗi JSONL thành công có tối đa một record cho một identity; errors.jsonl là nhật ký sự kiện và có thể có nhiều lần lỗi cho cùng sample. File lock ngăn hai process cùng ghi một run. Dòng cuối artifact output bị ghi dở do crash được backup thành `.torn-tail.<id>.bak` trước khi truncate; lỗi ở giữa output hoặc dòng JSON hỏng đã kết thúc bằng newline không bị bỏ qua. Quy tắc tự xóa JSON lỗi chỉ áp dụng cho **INPUT_JSONL**; generation source bên ngoài luôn được đọc nghiêm ngặt, không tự xóa/repair.

Resume bảo đảm không gọi lại **record thành công đã lưu**. Nếu kết nối timeout sau khi server đã xử lý, hoặc process chết giữa lúc nhận response và commit xuống đĩa, client không thể bảo đảm server chưa từng xử lý request đó; điều này cần idempotency ở endpoint. Pipeline gửi X-Request-ID nhất quán trong các HTTP retries của một lời gọi, nhưng không giả định server có hỗ trợ idempotency.

## Metric và contract kết quả

Tham khảo [bài báo LEGIT](https://aclanthology.org/2026.acl-long.150/), [repository LEGIT](https://github.com/jinulee-v/LEGIT) và [mã chấm reasoning task](https://github.com/jinulee-v/LEGIT/blob/master/reasoning_task_evaluate_legit.py). Giữ trọng số 5–2–3; dùng danh sách issues sẵn có, không dựng cây. Khác với pipeline gốc có rubric riêng cho conclusion toàn vụ, Final Order ở đây được suy ra từ **tập hợp phán quyết theo issues**, theo contract của benchmark này.

```text
N = số issues gốc của bản án
O = kết quả chung mà các phán quyết theo issues hàm ý có phù hợp tham chiếu không
c_i = có thực sự giải quyết issue i không (độc lập với đúng/sai)
k_i = phán quyết của issue đã covered có đúng/tương đương pháp lý không

final_order = 5 * O
coverage    = 2 * sum(c_i) / N
correctness = 3 * sum(c_i * k_i) / N
total       = final_order + coverage + correctness
```

N luôn là toàn bộ issues của sample. Không exact string matching để chấm nội dung pháp lý; chỉ đối chiếu nguyên văn original_issue để bảo vệ mapping. Không thêm điều kiện phải có conclusion chung. Nhãn `covered` và `correct` là boolean; issue không covered có `correct=false` khi tham chiếu xác định được.

JSON trả từ Judge tuân theo [judge_labels.schema.json](src/legal_application/schemas/judge_labels.schema.json), gồm doc_name, đủ issues_evaluation đúng thứ tự (issue_index bắt đầu từ 1, original_issue, covered, correct, extracted_model_verdict, inferred_reference_verdict, explanation, needs_review), final_order_correct, final_order_explanation, needs_review và evaluation_status. **Judge chỉ sinh nhãn**, không sinh score_components. Python tính và thêm score_components vào `evaluations.jsonl`; như vậy tổng điểm không do LLM quyết định. Ngoài JSON schema, kiểm tra cả số issues, nguyên văn issue, thứ tự, nhãn null/review và consistency.

Khi gold một issue không rõ, issue đó có needs_review=true, correct=null, inferred_reference_verdict=null và giải thích lý do. Cả record có evaluation_status=needs_review, final_order_correct=null, toàn bộ score_components=null. Không cho một issue mơ hồ điểm 0 giả và không giảm N. Record review bị loại khỏi **tất cả** trung bình và phân bố điểm, được đếm/list riêng. Nếu toàn run không có record chấm được, mean_score và các mean components là null.

## Artifacts và số đếm

Mỗi `OUTPUT_DIR/RUN_ID/` có:

| File | Nội dung |
|---|---|
| `generations.jsonl` | Một record cho mỗi generation thành công: identity, model, raw_response, status, sample/response checksum, usage/finish_reason và law sources |
| `evaluations.jsonl` | Một record cho mỗi lần Judge hoàn tất hợp lệ, gồm review: nhãn theo issue, score_components do Python tính, raw_judge_response, generator/judge model, mapping/checksum và retry metadata |
| `summary.json` | Trạng thái, số input/selected/attempted/completed/failed/skipped/pending/scored/review, mean 5/2/3/total, phân bố điểm, doc_names cần review/lỗi |
| `run.log` | Preflight, fallback nguồn luật, batch, API start/retry, structured retry, lưu từng record và tiến trình |
| `config.json` | Cấu hình thực tế, đường dẫn, endpoints/models, effort thực và hashes; không chứa API keys |
| `errors.jsonl` | Lỗi theo sample/stage/invocation, dùng phục hồi và thống kê cumulative failures |
| `removed_input_lines.jsonl` | Dòng JSON input bị loại: số dòng gốc, nguyên văn/base64, checksum, thời gian, path và nguyên nhân |
| `input_cleanup.json` | Các transaction cleanup, hash trước/sau, số dòng bị xóa và số dòng còn lại |

Các artifacts thành công không có record giả thay thế cho lỗi. Muốn đối chiếu toàn selected set, đọc thêm summary và errors.jsonl. Output có thể về khác thứ tự input do concurrency; join bằng **doc_name + input_index**, kèm sample checksum. Không zip theo vị trí dòng.

`counts.input` là tổng số dòng hợp lệ còn lại cộng số dòng đã xóa trong run; `input_valid` là số records validate được sau cleanup; `input_removed` là số dòng JSON lỗi đã xóa (giữ lịch sử khi resume). `selected` là min(LIMIT, input_valid), hoặc toàn bộ records hợp lệ khi LIMIT rỗng/all. Dòng đã xóa không bị tính thành sample failed hay điểm 0. `attempted` là số identity đã xử lý qua API hoặc được import/đã có lịch sử xử lý trong run; `completed` là số evaluations hợp lệ, gồm review; `failed` là số sample còn lỗi chưa khắc phục; `skipped` là số evaluations terminal có sẵn đầu invocation hiện tại; `scored` chỉ tính evaluations không review. `generation`/`evaluation` có số cumulative completed/failed/pending và `this_invocation` ghi attempted/completed/failed/skipped theo từng stage. Trong judge-only, import là generation skipped, không phải generator API attempted. Lỗi cũ được giữ trong errors.jsonl nhưng không tiếp tục tính failed khi sample đã hoàn thành.

Summary được ghi atomic sau mỗi batch và khi dừng có kiểm soát. Nếu bị kill cứng giữa batch, JSONL đã commit là nguồn chính xác để resume và tính lại summary. Nếu còn lỗi schema sau cleanup, summary có số dòng đã xóa, số records hợp lệ, danh sách validation errors và attempted=0; chưa gọi model khi schema chưa hợp lệ.

Mặc định mỗi batch có tối đa 20 samples: chạy tối đa 20 request generator đồng thời, rồi tối đa 20 request Judge đồng thời. Batch cuối ít hơn 20 vẫn được xử lý đầy đủ. LIMIT điều khiển tổng số samples của run, độc lập với BATCH_SIZE/concurrency. Mỗi stage có worker pool và semaphore riêng; không tạo số tasks theo toàn dataset. HTTP retry áp dụng cho timeout/connection lỗi, 408/409/429/5xx và malformed API envelope; backoff lũy tiến có jitter, tối đa khoảng 60 giây. Judge JSON/schema/mapping errors có budget retry riêng và luôn nhận đầy đủ dữ liệu ban đầu, không nối các draft lỗi vào context. HTTP 400/401/404/422 không retry ngầm. Input và các selected records được nạp vào RAM; BATCH_SIZE điều khiển lịch request, không phải một cam kết giới hạn RAM cho phần dữ liệu/artifacts đã nạp.

Exit codes: **0** khi hoàn tất hoặc validate thành công (có thể có review); **1** khi còn sample failed; **2** khi cấu hình/preflight/mapping bị lỗi; **130** khi Ctrl-C.

## Cấu hình

Mọi biến dưới đây có thể export hoặc sửa mặc định ở đầu script.

| Biến | Mặc định / ý nghĩa |
|---|---|
| `PROJECT_ROOT` | `/workspace/storage-shared/nlp/minhpn19/LegalApplication` |
| `INPUT_JSONL` | `$PROJECT_ROOT/input/judgment_with_issues_sample.jsonl` |
| `GENERATOR_BASE_URL`, `GENERATOR_MODEL` | Script điền sẵn `http://127.0.0.1:8001/v1`, `Qwen/Qwen3-4B`; sửa ở đầu script hoặc export để đổi model/server. Python CLI trực tiếp vẫn yêu cầu cấu hình rõ |
| `JUDGE_BASE_URL` | `http://172.29.62.173:8000/v1` |
| `JUDGE_MODEL` | `DeepSeek-V4.1-Flash-1017`; effort xhigh luôn giữ cố định |
| `LIMIT` | Rỗng/all/none = toàn bộ; 0 = không chọn sample; số dương = giới hạn chính xác |
| `BATCH_SIZE` | 20 |
| `GENERATOR_CONCURRENCY`, `JUDGE_CONCURRENCY` | 20, 20 |
| `MAX_OUTPUT_TOKENS` | 32768; cho phép 65536 |
| `OUTPUT_DIR` | `$PROJECT_ROOT/outputs` |
| `RUN_ID` | Timestamp giờ Việt Nam + hậu tố ngẫu nhiên nếu không đặt |
| `MODE` | full / judge-only / validate-only |
| `GENERATION_JSONL` | Path nguồn generation, bắt buộc ở judge-only |
| `RESUME` | 0; đặt 1 để tiếp tục run đã có |
| `API_TIMEOUT_SECONDS` | 1800 cho mỗi HTTP request |
| `API_MAX_RETRIES` | 3 retries ngoài lần HTTP đầu |
| `JUDGE_JSON_RETRIES` | 2 lần sửa structured response ngoài lần đầu |
| `RETRY_BACKOFF_SECONDS` | 2; tests đặt 0 |
| `GENERATOR_THINKING_MODE` | none / deepseek_vllm_xhigh |
| `JUDGE_RESPONSE_FORMAT` | json_schema / json_object, không tự fallback |
| `GENERATOR_API_KEY`, `JUDGE_API_KEY` | EMPTY nếu endpoint không yêu cầu key |

## Kiểm thử và giới hạn xác minh

```bash
python -m pytest -q
bash -n scripts/run_application.sh
python -m compileall -q src
```

Unit/integration tests dùng httpx.MockTransport; network thật bị chặn trong tests. Kiểm tra schema/fallback, chống leakage trong prompt, format từng issue, đầy đủ reference cho Judge, công thức, review/null, Final Order từ phán quyết từng issue, LIMIT >1000/0, batch cuối, concurrency và out-of-order mapping, retries, resume, crash tail, chống duplicate và artifacts JSONL. Tests cleanup kiểm tra lỗi giữa file, lỗi ngoài LIMIT, UTF-8 lỗi, giữ nguyên bytes của dòng tốt, log khôi phục được dữ liệu gốc, không xóa schema lỗi, lỗi ghi journal, phục hồi transaction gián đoạn và resume không sinh trùng. Tests không xác minh chất lượng phán đoán pháp lý của Judge thật, khả năng endpoint hỗ trợ json_schema + reasoning parser, tên checkpoint generator, context window hay throughput trên B200. Cần kiểm tra các điểm đó bằng smoke run 1–5 samples trên runtime được chỉ định trước khi chạy full.
