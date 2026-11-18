# Legal Application Benchmark

Hướng dẫn đầy đủ từng lệnh với checkpoint local Qwen3-4B, môi trường Python hiện có và concurrency 20: [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md).

Đánh giá năng lực áp dụng pháp luật của LLM trên bản án Việt Nam. Mỗi bản án được gửi trong **một request generator**, gồm toàn bộ facts_clean, issues và luật theo thứ tự gốc. Bài làm là văn bản tự do, giải quyết và đưa ra phán quyết **riêng cho từng issue**. Không yêu cầu phán quyết cuối cùng cho toàn vụ án.

Judge đọc đầy đủ nguồn tham chiếu, suy ra kết quả từng issue trực tiếp từ application/conclusion, gán nhãn và giải thích. Python kiểm tra schema, mapping và tính điểm. Không xây Legal Issue Tree hoặc một pipeline tạo gold riêng.

## Data Completeness Classification và input có nhãn

Chạy `scripts/statistic.sh` trên JSONL gốc **chưa có data_status** để phân loại độ đầy đủ của facts_clean theo tất cả issues. Judge đọc facts_clean, issues, application/conclusion và facts riêng để kiểm chứng. `full` nghĩa dữ kiện thực tế đủ để giải quyết mọi issue; `partial` nghĩa ít nhất một issue thiếu tình tiết/chứng cứ quyết định được Tòa sử dụng. Diễn đạt tương đương hoặc suy ra hợp lý vẫn được coi là đã có dữ kiện. Thiếu lập luận pháp luật/trích dẫn luật không phải thiếu facts. Prompt phân loại nằm riêng, không sửa prompts Application hay công thức LEGIT.

Statistic **không gọi clean_input và không sửa input gốc**. JSON/schema lỗi (kể cả issues rỗng như dòng 578) được báo warning, lưu riêng vào invalid_input_records.jsonl/errors.jsonl và **bỏ qua để tiếp tục phân loại các records hợp lệ phía sau**. LIMIT chỉ tính records hợp lệ; số dòng gốc không bị đánh lại trong classification_details. Nếu không còn record hợp lệ thì báo lỗi và không gọi Judge. Nếu input đã có data_status, dùng bản input gốc chưa gắn nhãn để tránh ghi đè giá trị cũ.

```bash
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
cd "$PROJECT_ROOT"
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export STAT_RUN_ID="completeness-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

INPUT_JSONL="$PROJECT_ROOT/input/judgment_with_issues_sample.jsonl" \
  RUN_ID="$STAT_RUN_ID" LIMIT=all BATCH_SIZE=20 JUDGE_CONCURRENCY=20 RESUME=0 \
  JUDGE_MODEL=DeepSeek-V4.1-Flash-1017 JUDGE_BASE_URL=http://172.29.62.173:8000/v1 \
  JUDGE_THINKING_MODE=deepseek_vllm_xhigh bash scripts/statistic.sh

python -m json.tool "$OUTPUT_DIR/$STAT_RUN_ID/summary.json"
```

Artifacts: `classification_details.jsonl` có lý do, phân tích đủ issues, facts thiếu, trích dẫn nguồn, raw Judge response và metadata; `summary.json` có input/input_valid/input_invalid/selected/processed/full/partial/failed, tỷ lệ và doc_names; `errors.jsonl` có lỗi; `invalid_input_records.jsonl` giữ nguyên dòng nguồn bị bỏ qua dưới dạng text/base64 cùng số dòng, doc_name, lý do và checksum; `run.log` ghi tiến trình. Các quotes phải là substring nguyên văn của đúng trường nguồn; mapping, nhãn và tính nhất quán full/partial được validate và retry. Kiểm tra quote không thay thế đánh giá tương đương ngữ nghĩa do Judge thực hiện.

Chỉ khi **toàn bộ selected samples hợp lệ** phân loại thành công, pipeline mới xuất `classified_input.jsonl` theo thứ tự input, giữ nguyên mọi field/value của các records đó và chỉ thêm data_status. Record input lỗi không được xuất hoặc gán nhãn giả. Nếu còn lỗi Judge, các kết quả thành công nằm trong `classified_input.partial.jsonl`, `classified_output_complete=false`. Lỗi API/không phân loại được không bao giờ được gán partial. Tỷ lệ full/partial tính trên processed thành công; tỷ lệ failed tính trên selected; tỷ lệ input_invalid tính riêng trên tổng dòng input. LIMIT rỗng/all không có hidden cap; LIMIT hữu hạn xuất đúng selected subset. `classified_output_complete=true` nghĩa hoàn tất tập **selected_valid_records**, không chứng nhận các dòng input lỗi; `source_fully_classified` chỉ true khi toàn bộ file nguồn đều hợp lệ và đã phân loại xong. Run hoàn tất với dòng nguồn bị bỏ qua có status=completed_with_input_errors và vẫn xuất kết quả của records hợp lệ.

Resume statistic với **cùng cấu hình, input và RUN_ID**; chỉ gọi lại samples chưa thành công:

```bash
INPUT_JSONL="$PROJECT_ROOT/input/judgment_with_issues_sample.jsonl" \
  RUN_ID="$STAT_RUN_ID" LIMIT=all BATCH_SIZE=20 JUDGE_CONCURRENCY=20 RESUME=1 \
  JUDGE_MODEL=DeepSeek-V4.1-Flash-1017 JUDGE_BASE_URL=http://172.29.62.173:8000/v1 \
  JUDGE_THINKING_MODE=deepseek_vllm_xhigh bash scripts/statistic.sh
```

Các lệnh resume/relocate dưới đây dùng khi code/prompt/schema còn khớp run gốc. Nếu cập nhật từ phiên bản cũ lên bản sửa retry hiện tại, dùng retry_statistic.sh để nhập các kết quả thành công vào run mới như bên dưới.

Muốn ngắt để tăng batch/concurrency mà vẫn giữ kết quả statistic cũ: Ctrl-C và đợi process thoát, giữ RUN_ID đang chạy, rồi dùng helper. Ví dụ:

```bash
# Thay TEN_RUN_DANG_CHAY bằng RUN_ID thật đã có trong outputs.
bash scripts/resume_statistic.sh TEN_RUN_DANG_CHAY \
  --batch-size 200 --judge-concurrency 200
```

Helper khôi phục input, LIMIT, Judge, thinking, output budget và các thiết lập còn lại từ config.json; cập nhật batch/concurrency đã chỉ định, backup cấu hình cũ thành config.before-resume-*.json và ghi resume_config_changes.jsonl. Không sửa input/results, không gọi lại records thành công. Nó từ chối khi process cũ còn giữ lock hoặc input/code khác run gốc. Hai helper nằm ngoài package nên việc chỉ cập nhật chúng không làm đổi implementation hash. Cơ chế này dành riêng cho statistic; resume thông thường vẫn kiểm tra snapshot nghiêm ngặt.

Nếu đã **chuyển project và outputs sang path mới**, đồng bộ code mới lên B200, đặt trực tiếp PROJECT_ROOT/OUTPUT_DIR mới và dùng `--relocate`. Đừng lấy PROJECT_ROOT từ config.json cũ để `cd`, vì file đó vẫn chứa path trước khi di chuyển. Với run `20261117T135710_be74ac50`:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
cd "$PROJECT_ROOT"
bash scripts/resume_statistic.sh 20261117T135710_be74ac50 \
  --relocate --batch-size 200 --judge-concurrency 200
```

`--relocate` cập nhật project_root/output_dir trong config đã sao lưu. Input nằm trong project cũ được đổi sang vị trí tương ứng trong project mới; input bên ngoài, như `/workspace/storage-shared/nlp/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl`, giữ nguyên path. Nếu input cũng được di chuyển riêng, thêm `--input-jsonl /duong/dan/moi/input.jsonl`; checksum phải khớp tuyệt đối run cũ. Helper chỉ chấp nhận thay đổi implementation hash khi xác minh code chỉ khác đúng giá trị path mặc định trong Config; các thay đổi code/prompt/schema khác vẫn bị từ chối. Cấu hình và nhãn Judge cũ được giữ để đối chiếu trong backup/audit.

Application luôn chỉ chọn samples **full**, lọc trước LIMIT; không còn tùy chọn DATA_STATUS_USED. Có thể đọc file có nhãn full/partial của statistic (partial tự bị bỏ qua), hoặc dùng script dưới để xuất input giữ nguyên cấu trúc và bytes nguồn, kể cả khi run statistic còn sample failed:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
cd "$PROJECT_ROOT"
bash scripts/filter_full.sh \
  --input /workspace/storage-shared/nlp/users/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl \
  --classifications "$PROJECT_ROOT/outputs/20261117T202530_5e87eebe/classification_details.jsonl" \
  --output "$PROJECT_ROOT/input/input_application_clean_v1.jsonl"

INPUT_JSONL="$PROJECT_ROOT/input/input_application_clean_v1.jsonl" \
  MODE=full LIMIT=all RESUME=0 RUN_ID="application-full-${RANDOM}" \
  bash scripts/run_application.sh
```

Script nối bằng document_id nếu có; artifact cũ dùng doc_name được nối theo doc_name. Khi cả hai có mặt phải nhất quán; sample_sha256 được kiểm tra nếu artifact có lưu. Chỉ nhãn completed/full được xuất, theo thứ tự nguồn; không thêm data_status hoặc serialize lại JSON. File input_application_clean_v1.jsonl.manifest.json ghi checksum và danh sách full để Application xác minh input gốc không có nhãn. Giữ manifest cạnh output; không sửa nội dung file sau khi lọc. Dòng nguồn JSON/schema lỗi được bỏ qua, báo vào input_application_clean_v1.jsonl.filter_log.jsonl. File nguồn và classification_details không bị sửa. Có thể lặp --classifications để kết hợp nhiều run; nhãn xung đột bị từ chối.

Record lỗi đã cung cấp có finish_reason=stop ở cả ba lần, nhưng validation_errors báo quote không khớp facts_clean/application: đó là lỗi sao chép evidence, không phải trường hợp bị cắt output. Không suy ra mọi sample failed đều cùng nguyên nhân nếu chưa thống kê.

Statistic dùng evidence_format=source_ids_v1 cho request mới: toàn bộ nguồn vẫn được gửi, kèm catalog chia thành các đoạn liên tục có ID. Judge chọn quote_id cho application/conclusion và các ID trong facts_clean_evidence/raw_facts_evidence; JSON Schema giới hạn các lựa chọn vào ID đã có. Python tra ID, kiểm tra đúng trường nguồn và lấy nguyên văn đoạn gốc để ghi quote trong classification_details. Không bắt model chép lại văn bản nguồn, không fuzzy matching và không tự đổi nhãn để vượt validation. ID không tồn tại/dùng nhầm nguồn vẫn là lỗi. Catalog bao phủ toàn bộ nguồn; 500 ký tự là kích thước mỗi đoạn, không phải giới hạn/cắt input.

raw_judge_response giữ nguyên JSON chứa IDs; metadata.normalizations ghi các ID và vị trí start/end đã mở rộng. Artifact ngoài vẫn có cấu trúc evidence.quote và các đoạn trích như trước; seed/resume kiểm chứng lại được cả giao thức ID mới và quote cũ. Với quote cũ chỉ có thể sửa JSON fence/khoảng trắng rồi validate nguyên văn. Retry giữ response trước đó và vị trí field lỗi. errors.jsonl lưu last_validation_error, invalid_responses và usage/finish_reason; summary có failure_reasons. Không tự thay đổi thinking hoặc tăng token budget.

Sau khi cập nhật code này, dùng **run mới có nhập kết quả cũ** để tránh xung đột implementation hash. Run cũ giữ nguyên, các nhãn thành công được kiểm tra lại identity/hash/schema/evidence/model trước khi tái sử dụng:

```bash
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export RETRY_RUN_ID="retry-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"
bash scripts/retry_statistic.sh 20261117T202530_5e87eebe \
  --run-id "$RETRY_RUN_ID" \
  --input-jsonl /workspace/storage-shared/nlp/users/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl \
  --batch-size 200 --judge-concurrency 200 \
  --judge-json-retries 4 --max-output-tokens 65536
```

Với log đã nêu, 810 kết quả cũ được nhập, chỉ 1.199 samples còn thiếu được gọi Judge. --judge-json-retries 4 nghĩa tối đa 5 lần trả structured output cho mỗi sample. Input phải giữ đúng checksum của run cũ, kể cả khi path đã đổi. Judge/thinking/LIMIT/timeout được khôi phục từ config cũ; có thể chỉ định --judge-thinking-mode nếu muốn đổi rõ ràng cho các request mới. Seed và thống kê nguyên nhân lỗi cũ nằm ở outputs/.statistic-seeds/$RETRY_RUN_ID/source.json. Lọc lại bằng --classifications "$OUTPUT_DIR/$RETRY_RUN_ID/classification_details.jsonl" để lấy cả full cũ và full mới. Resume run mới bằng scripts/resume_statistic.sh "$RETRY_RUN_ID"; helper resume không bỏ qua thay đổi code/prompt/schema của run cũ.

## Cài đặt trên B200

Client cần Python **>=3.10**, không cần GPU hay PyTorch. Dùng môi trường `.venv` hiện có của bạn, đặt ở bất kỳ thư mục nào. Script gọi `python` trên PATH sau khi activate môi trường; không kiểm tra phiên bản thư viện hoặc yêu cầu `.venv` trong project. Server vLLM được triển khai riêng.

```bash
# Activate môi trường .venv hiện có của bạn trước khi chạy các lệnh này.
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/storage-shared/nlp/users/minhpn19/LegalApplication}"
cd "$PROJECT_ROOT"
```

Nếu môi trường đã có đầy đủ dependencies, chạy trực tiếp entry point. Khi cần cài vào một môi trường mới, dùng `python -m pip install -r requirements.txt`; `requirements-dev.txt` bao gồm dependencies runtime và pytest. Script chạy trực tiếp package từ `src/`; nếu muốn dùng thêm lệnh `legal-application` sau khi cài package, có thể cài `python -m pip install -e .`. Dependencies cũng được khai báo trong `pyproject.toml`. vLLM được cài riêng trong môi trường host model.

Entry point chính: `bash "$PROJECT_ROOT/scripts/run_application.sh"`. Script không tự cài hoặc khởi động model. Paths tương đối được resolve từ PROJECT_ROOT, kể cả khi gọi từ thư mục khác. Path tuyệt đối do người dùng export được giữ nguyên.

Terminal có thanh tiến trình `tqdm`: **Scan input** và **Validate input** theo số bytes; **Generation** và **Judge** theo số samples của toàn run, có số batch, thời gian, tốc độ và ETA. Postfix cho biết `ok`, `failed`, `skipped`, `review` và `blocked` (Judge không được gọi vì sample chưa có generation). 100% nghĩa đã xử lý hết samples, không đồng nghĩa tất cả thành công. Resume khởi tạo thanh từ số kết quả đã lưu, không cộng trùng; judge-only hiển thị Import generations thay cho Generation. Khi stderr không phải terminal, thanh tự ẩn và vẫn ghi đầy đủ run.log. Dependencies tqdm nằm trong requirements.txt/pyproject.toml.

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
| `data_status` | Nếu có phải là full/partial; Application chỉ chọn full. Input raw không có nhãn cần manifest từ filter_full.sh |

Fallback chỉ áp dụng ở **cấp luật khi không có provisions**; không tự thay context_text của provision bằng exact_text. Nguồn fallback được ghi trong prompt, `metadata.law_sources` của generation và run.log. Không chọn lọc luật theo mức liên quan, không cắt ngắn hoặc chuẩn hóa lại nội dung. Generator dùng projection allowlist, không nhận application, conclusion, markdown, explain hoặc metadata của luật chứa nguồn đáp án. Judge nhận toàn bộ benchmark_seed, issues, luật gốc và nguyên văn message.content của generator.

Riêng Application giữ cơ chế cleanup hiện có: sau khi kiểm tra data_status, pipeline quét **toàn bộ file**, kể cả phần ngoài LIMIT. Dòng không parse được strict JSON/UTF-8 (kể cả dòng trắng, key JSON trùng và NaN/Infinity), record sai schema nền (ví dụ issues thiếu/rỗng/sai kiểu hoặc thiếu nguồn luật) và doc_name trùng được ghi đầy đủ vào `OUTPUT_DIR/RUN_ID/removed_input_lines.jsonl`, sau đó **xóa vĩnh viễn khỏi chính INPUT_JSONL**. Pipeline tiếp tục với các dòng hợp lệ phía sau; những lần chạy sau đọc file đã làm sạch nên không gặp lại các dòng đã xóa. LIMIT được áp dụng lên các samples còn lại sau khi lọc data_status, không tính các dòng bị xóa vào quota. Với doc_name trùng, giữ record hợp lệ đầu tiên và loại các bản xuất hiện sau. Statistic không sử dụng cơ chế cleanup này.

Log có đường dẫn input, thời gian, số dòng/index gốc, doc_name nếu đọc được, loại lỗi/lý do/vị trí lỗi, nguyên văn dòng, bản base64 của toàn bộ bytes và checksum. `validation_stage` phân biệt json / schema / duplicate_doc_name. `input_cleanup.json` lưu hash trước/sau, số dòng đã xóa, removal_counts theo loại lỗi và trạng thái transaction. Journal được ghi atomic và fsync **trước khi** thay input bằng file sạch; nếu ghi log thất bại, input được giữ nguyên và chưa gọi model. File sạch giữ nguyên bytes và thứ tự của các dòng còn lại. Lock riêng ở `<INPUT_JSONL>.cleanup.lock` ngăn hai process đồng thời làm sạch cùng file.

Cleanup kiểm tra cả schema; sau đó đọc và kiểm tra lại file sạch trước khi gọi model. `input_index` là số dòng vật lý **bắt đầu từ 0 của file đã làm sạch**; `input_line` bắt đầu từ 1. Vì thế index ổn định khi resume hoặc chạy lại file sạch; số dòng trước khi xóa nằm trong removal log. Pipeline không tự tạo issues, facts hoặc luật để bù cho record lỗi. Nếu không còn sample nào sau cleanup, pipeline lưu log và báo lỗi input rỗng.

Hành vi làm sạch áp dụng cho cả `full`, `judge-only` và `validate-only`. `validate-only` không gọi model nhưng **có sửa file input** khi phát hiện JSON lỗi. Ví dụ:

```bash
MODE=validate-only RUN_ID=schema-check \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Tệp mẫu gốc trong repository có 6 records hợp lệ nhưng chưa có data_status; cần phân loại bằng statistic trước. Bản raw có JSON/schema lỗi sẽ được statistic báo cáo, bỏ qua và giữ nguyên nguồn; Application vẫn có cleanup như mô tả trên sau khi input có nhãn hợp lệ. Xem `counts.input_invalid` của statistic để biết số dòng nguồn bị bỏ qua; `counts.input_removed` của Application là số dòng bị xóa ở cơ chế cleanup riêng. Để smoke test 5 bản án, giữ nguyên file đã lọc cùng manifest và dùng LIMIT=5; không cắt file bằng head vì sẽ làm lệch checksum:

```bash
MODE=validate-only LIMIT=5 INPUT_JSONL=input/input_application_clean_v1.jsonl RUN_ID=smoke-schema \
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

Judge mặc định là `DeepSeek-V4.1-Flash-1017` tại `http://172.29.62.173:8000/v1`, với `JUDGE_THINKING_MODE=deepseek_vllm_xhigh`. Cách gửi được đối chiếu trực tiếp với `LegalDataset/src/legal_issue_pipeline/llm/adapters.py`, `DeepSeekVLLMAdapter` và `tests/unit/test_vllm_effort.py` trong repository lân cận:

```json
{"chat_template_kwargs":{"thinking":true,"reasoning_effort":75}}
```

Không gửi `reasoning_effort="xhigh"` ở top level và không dùng cách map của hosted DeepSeek API. Budget 75 là tham số effort, độc lập với `max_tokens`. Mặc định generator không gửi tham số reasoning riêng và dùng thiết lập của server/template. Nếu generator cũng là DeepSeek vLLM tương ứng, đặt `GENERATOR_THINKING_MODE=deepseek_vllm_xhigh` để gửi đúng body trên.

Hai vai trò có thinking mode riêng, có thể export hoặc sửa mặc định trong script:

```bash
export GENERATOR_THINKING_MODE=none
export JUDGE_THINKING_MODE=deepseek_vllm_xhigh
```

Các lựa chọn dưới đây dùng được độc lập cho cả GENERATOR_THINKING_MODE và JUDGE_THINKING_MODE:

| Giá trị | Body gửi tới endpoint |
|---|---|
| `low` | DeepSeek vLLM: thinking=true, reasoning_effort=25 |
| `high` | DeepSeek vLLM: thinking=true, reasoning_effort=50 |
| `xhigh` | DeepSeek vLLM: thinking=true, reasoning_effort=75 |
| `max` | DeepSeek vLLM: thinking=true, reasoning_effort=100 |
| Số nguyên `1`–`100`, ví dụ `63` | DeepSeek vLLM: thinking=true, reasoning_effort là số nguyên tương ứng |
| `on` / `off` | DeepSeek vLLM: thinking=true / false, không gửi effort |
| `default` | Không gửi tham số thinking/effort, dùng server/template |
| `none` | Alias cũ của default, giữ tương thích; KHÔNG có nghĩa tắt thinking |
| `qwen3_on` / `qwen3_off` | Qwen3: enable_thinking=true / false, không gửi effort DeepSeek |
| `deepseek_vllm_<giá trị>` | Alias rõ backend, ví dụ deepseek_vllm_high, deepseek_vllm_max, deepseek_vllm_63, deepseek_vllm_off |

Các tham số template trong bảng được gửi bên trong `chat_template_kwargs`. Muốn tắt DeepSeek, dùng `off`; muốn tắt Qwen3, dùng `qwen3_off`. `medium` và `minimal` không được hỗ trợ bởi DeepSeek V4.1 vLLM; pipeline từ chối rõ thay vì tự map. Theo [recipe chính thức](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash#reasoning-and-tool-calling), backend này có bốn tier 25/50/75/100 và budget nguyên 1–100. [DeepSeek hosted API](https://api-docs.deepseek.com/api/create-chat-completion/) có interface và tier mapping khác; các mức ở đây dành cho endpoint vLLM, không phải adapter API hosted. Budget effort không phải số output tokens; MAX_OUTPUT_TOKENS vẫn độc lập.

Ví dụ chọn hai mức khác nhau hoặc chọn budget cụ thể:

```bash
export GENERATOR_THINKING_MODE=high
export JUDGE_THINKING_MODE=max

# Hoặc đặt Judge ở budget 63:
export JUDGE_THINKING_MODE=63
```

Mode yêu cầu, backend, trạng thái bật/tắt, budget và tham số thực tế của cả hai vai trò được ghi trong config.json và run.log; metadata từng request ghi actual_thinking_parameters. Đổi mode cần RUN_ID mới. Mặc định cũ vẫn giữ: generator dùng none, Judge dùng deepseek_vllm_xhigh.

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

`counts.input` là tổng số dòng hợp lệ còn lại cộng số dòng đã xóa trong run; `input_valid` là số records validate được sau cleanup; `input_removed` là số dòng JSON lỗi đã xóa (giữ lịch sử khi resume). `selected` là min(LIMIT, số records full hợp lệ), hoặc toàn bộ records full khi LIMIT rỗng/all. Dòng đã xóa không bị tính thành sample failed hay điểm 0. `attempted` là số identity đã xử lý qua API hoặc được import/đã có lịch sử xử lý trong run; `completed` là số evaluations hợp lệ, gồm review; `failed` là số sample còn lỗi chưa khắc phục; `skipped` là số evaluations terminal có sẵn đầu invocation hiện tại; `scored` chỉ tính evaluations không review. `generation`/`evaluation` có số cumulative completed/failed/pending và `this_invocation` ghi attempted/completed/failed/skipped theo từng stage. Trong judge-only, import là generation skipped, không phải generator API attempted. Lỗi cũ được giữ trong errors.jsonl nhưng không tiếp tục tính failed khi sample đã hoàn thành.

Summary được ghi atomic sau mỗi batch và khi dừng có kiểm soát. Nếu bị kill cứng giữa batch, JSONL đã commit là nguồn chính xác để resume và tính lại summary. Cleanup loại record JSON/schema lỗi; nếu không còn record hoặc input thay đổi ngoài tiến trình trong lúc kiểm tra, preflight vẫn dừng rõ ràng trước mọi request.

Mặc định mỗi batch có tối đa 20 samples: chạy tối đa 20 request generator đồng thời, rồi tối đa 20 request Judge đồng thời. Batch cuối ít hơn 20 vẫn được xử lý đầy đủ. LIMIT điều khiển tổng số samples của run, độc lập với BATCH_SIZE/concurrency. Mỗi stage có worker pool và semaphore riêng; không tạo số tasks theo toàn dataset. HTTP retry áp dụng cho timeout/connection lỗi, 408/409/429/5xx và malformed API envelope; backoff lũy tiến có jitter, tối đa khoảng 60 giây. Judge JSON/schema/mapping errors có budget retry riêng và luôn nhận đầy đủ dữ liệu ban đầu, không nối các draft lỗi vào context. HTTP 400/401/404/422 không retry ngầm. Input và các selected records được nạp vào RAM; BATCH_SIZE điều khiển lịch request, không phải một cam kết giới hạn RAM cho phần dữ liệu/artifacts đã nạp.

Exit codes: **0** khi hoàn tất hoặc validate thành công (có thể có review); **1** khi còn sample failed; **2** khi cấu hình/preflight/mapping bị lỗi; **130** khi Ctrl-C.

## Cấu hình

Mọi biến dưới đây có thể export hoặc sửa mặc định ở đầu script.

| Biến | Mặc định / ý nghĩa |
|---|---|
| `PROJECT_ROOT` | `/workspace/storage-shared/nlp/users/minhpn19/LegalApplication` |
| `INPUT_JSONL` | Application: `$PROJECT_ROOT/input/input_application_clean_v1.jsonl`; statistic: input raw |
| `GENERATOR_BASE_URL`, `GENERATOR_MODEL` | Script điền sẵn `http://127.0.0.1:8001/v1`, `Qwen/Qwen3-4B`; sửa ở đầu script hoặc export để đổi model/server. Python CLI trực tiếp vẫn yêu cầu cấu hình rõ |
| `JUDGE_BASE_URL` | `http://172.29.62.173:8000/v1` |
| `JUDGE_MODEL` | `DeepSeek-V4.1-Flash-1017` |
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
| `GENERATOR_THINKING_MODE` | Mặc định none; hỗ trợ default/none, low/high/xhigh/max, số nguyên 1–100, on/off, alias deepseek_vllm_*, qwen3_on/off |
| `JUDGE_THINKING_MODE` | Mặc định deepseek_vllm_xhigh; cùng toàn bộ lựa chọn với generator |
| `JUDGE_RESPONSE_FORMAT` | json_schema / json_object, không tự fallback |
| `GENERATOR_API_KEY`, `JUDGE_API_KEY` | EMPTY nếu endpoint không yêu cầu key |

## Kiểm thử và giới hạn xác minh

```bash
python -m pytest -q
bash -n scripts/run_application.sh
python -m compileall -q src
```

Unit/integration tests dùng httpx.MockTransport; network thật bị chặn trong tests. Kiểm tra schema/fallback, chống leakage trong prompt, format từng issue, đầy đủ reference cho Judge, công thức, review/null, Final Order từ phán quyết từng issue, LIMIT >1000/0, batch cuối, concurrency và out-of-order mapping, retries, resume, crash tail, chống duplicate và artifacts JSONL. Tests cleanup kiểm tra lỗi giữa file, lỗi ngoài LIMIT, UTF-8 lỗi, giữ nguyên bytes của dòng tốt, log khôi phục được dữ liệu gốc, loại record schema lỗi có đầy đủ log, tái hiện issues rỗng ở dòng 578, lỗi ghi journal, phục hồi transaction gián đoạn và resume không sinh trùng. Tests không xác minh chất lượng phán đoán pháp lý của Judge thật, khả năng endpoint hỗ trợ json_schema + reasoning parser, tên checkpoint generator, context window hay throughput trên B200. Cần kiểm tra các điểm đó bằng smoke run 1–5 samples trên runtime được chỉ định trước khi chạy full.
