# Hướng dẫn chạy Legal Application Benchmark trên B200

## Rule Application theo rubric

Chạy các lệnh trong môi trường Python đã activate. `generate_rubrics.sh` gọi endpoint DeepSeek có sẵn tại `http://172.29.116.48:8000/v1`, mặc định dùng `DeepSeek-V4.1-Flash-1017` và `deepseek_vllm_xhigh` (effort 75). Sửa `RUBRIC_BASE_URL`, `RUBRIC_MODEL`, `RUBRIC_API_KEY`, `RUBRIC_THINKING_MODE`, `RUBRIC_CONCURRENCY` hoặc `RUBRIC_MAX_OUTPUT_TOKENS` qua biến môi trường khi cần. `RUBRIC_RESPONSE_FORMAT=json_object` dùng khi endpoint chỉ hỗ trợ chế độ JSON object. Python vẫn kiểm tra cùng schema.

### Thử 10 mẫu trên RTX 3090

```bash
export PROJECT_ROOT=/mnt/hdd/nhatminh/LegalApplication
cd "$PROJECT_ROOT"

RAW_INPUT_JSONL="$PROJECT_ROOT/input/sample_v2.jsonl" \
  FINAL_APPLICATION_JSONL="$PROJECT_ROOT/outputs/rubric-smoke/application.jsonl" \
  FINAL_ISSUE_SPOTTING_JSONL="$PROJECT_ROOT/outputs/rubric-smoke/issue_spotting.jsonl" \
  EXPECTED_COUNT=10 LIMIT=10 BATCH_SIZE=2 RUBRIC_CONCURRENCY=2 \
  RUN_ID=rubric-smoke-10 RESUME=0 bash scripts/generate_rubrics.sh
```

### Sinh và kiểm tra 600 mẫu trên B200

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export FINAL_APPLICATION_JSONL="$PROJECT_ROOT/input/final/final_600_rule_application.jsonl"
export FINAL_ISSUE_SPOTTING_JSONL="$PROJECT_ROOT/input/final/final_600_issue_spotting.jsonl"
cd "$PROJECT_ROOT"

RAW_INPUT_JSONL="$PROJECT_ROOT/input/final_raw_600_input_application.jsonl" \
  EXPECTED_COUNT=600 LIMIT=all BATCH_SIZE=20 RUBRIC_CONCURRENCY=20 \
  RUN_ID=rubric-600-v1 RESUME=0 bash scripts/generate_rubrics.sh

EVALUATION_PROFILE=rubric INPUT_JSONL="$FINAL_APPLICATION_JSONL" \
  MODE=validate-only EXPECTED_COUNT=600 LIMIT=all RUN_ID=validate-rubric-600-v1 \
  RESUME=0 bash scripts/run_application.sh
```

Script kiểm tra toàn bộ file nguồn, kể cả các dòng ngoài `LIMIT`. Với `LIMIT` lớn hơn 0 và nhỏ hơn tổng số dòng, kết quả đủ của tập con được ghi dưới run directory với hậu tố `_subset.jsonl`. Hai file final cùng manifest chỉ được xuất với đủ số mẫu yêu cầu và rubric của mọi mẫu đã qua kiểm định. Script từ chối ghi đè nếu một file đích đã tồn tại với nội dung khác. Khi xuất thành công, manifest nằm cạnh Application với hậu tố `.manifest.json`, có checksum của cả hai file và đối chiếu `doc_name` theo dòng gốc. Để đánh giá một phiên bản đã xuất, dùng file Application và manifest đi cùng.

Dùng `LIMIT=0` để kiểm tra riêng dữ liệu raw trước khi gọi API.

```bash
EXPECTED_COUNT=600 LIMIT=0 RUN_ID=raw-rubric-preflight RESUME=0 bash scripts/generate_rubrics.sh
```

Khi input hợp lệ, kết quả có `status=input_validated` và `counts.valid=600`. Chi tiết lỗi theo dòng và thống kê theo nguyên nhân nằm trong `rubric_generation_summary.json`. Nhãn `Commercial` dùng taxonomy riêng cho án kinh doanh thương mại. Nếu run cũ dừng ngay ở validation, chọn RUN_ID mới cho lần sinh tiếp theo vì run cũ chưa có snapshot để resume.

### Chạy đánh giá

`GENERATOR_BASE_URL` và `GENERATOR_MODEL` phải khớp endpoint và served model bạn muốn đánh giá. Judge có các biến `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY`, `JUDGE_THINKING_MODE` và `JUDGE_RESPONSE_FORMAT` riêng. Ví dụ với Qwen3-4B đã host ở port 8001:

```bash
GENERATOR_BASE_URL=http://127.0.0.1:8001/v1 GENERATOR_MODEL=Qwen/Qwen3-4B \
  JUDGE_BASE_URL=http://172.29.116.48:8000/v1 JUDGE_MODEL=DeepSeek-V4.1-Flash-1017 \
  EVALUATION_PROFILE=rubric MODE=full EXPECTED_COUNT=600 LIMIT=all \
  INPUT_JSONL="$FINAL_APPLICATION_JSONL" RUN_ID=evaluate-qwen3-4b-rubric-v1 \
  RESUME=0 bash scripts/run_application.sh
```

Chấm lại cùng câu trả lời bằng Judge khác trong một run riêng:

```bash
EVALUATION_PROFILE=rubric MODE=judge-only EXPECTED_COUNT=600 LIMIT=all \
  INPUT_JSONL="$FINAL_APPLICATION_JSONL" \
  GENERATION_JSONL="$OUTPUT_DIR/evaluate-qwen3-4b-rubric-v1/generations.jsonl" \
  RUN_ID=rejudge-rubric-v1 RESUME=0 bash scripts/run_application.sh
```

### Tiếp tục run và đọc báo cáo

Dùng đúng run ID trong `OUTPUT_DIR`. Helper khôi phục cấu hình đã lưu và tiếp tục các sample chưa hoàn thành. Nếu API cần xác thực, export lại API key trong terminal hiện tại.

```bash
bash scripts/resume_rubric_run.sh rubric-600-v1
# Hoặc tiếp tục run đánh giá
bash scripts/resume_rubric_run.sh evaluate-qwen3-4b-rubric-v1
```

`config.json` cố định code, input, endpoint, model, thinking, lịch batch, giới hạn và các cài đặt của run. Sau khi đổi code hoặc rubric, dùng run ID mới. Các run statistic trước refactor cần bản code tương ứng hoặc nhập lại kết quả hợp lệ qua `retry_statistic.sh`.

Mỗi sample qua ba lượt gọi rubric gồm phân rã nhận định, ánh xạ vào issues, rồi kiểm định ngữ nghĩa từ nguồn. Bước Python xác minh evidence nguyên văn, đơn vị nguồn, ID, thứ tự và phạm vi. Các nhóm nhận định tham khảo [Điều 260 Bộ luật Tố tụng hình sự 2015](https://vanban.chinhphu.vn/default.aspx?docid=183217&pageid=27160), [Điều 266 Bộ luật Tố tụng dân sự 2015](https://vanban.chinhphu.vn/?classid=1&docid=183189&pageid=27160&typegroupid=3) và [Điều 194 Luật Tố tụng hành chính 2015](https://vanban.chinhphu.vn/default.aspx?docid=183190&pageid=27160). Category và cấp xét xử dùng trong quá trình sinh rubric. Cả năm category đã xuất hiện trong file mẫu. Category chưa hỗ trợ gây lỗi validation có số dòng.

Trong run sinh rubric, xem `rubric_generation_summary.json` để biết số mẫu hoàn thành, lỗi và những `doc_name` còn thiếu. `rubric_coverage_report.json` báo tỷ lệ theo các unit thuộc phạm vi mà lượt audit đã chấp nhận, kèm số mẫu chưa được kiểm định thành công. `source_units.jsonl`, `added_issues.jsonl` và `excluded_units.jsonl` lưu nguồn và quyết định ánh xạ. `rubric_review.jsonl` lưu thiếu hụt input, verdict chưa xác định và lỗi sinh rubric. Issue gốc chưa có nhận định hỗ trợ được giữ cùng lý do review và danh sách nhóm rỗng.

Trong run đánh giá, `generations.jsonl` lưu nguyên văn câu trả lời. `evaluations.jsonl` lưu từng criterion, bằng chứng trích từ câu trả lời, điểm issue và Outcome Accuracy. Application Score lấy trung bình các issue đủ điều kiện, rồi lấy trung bình các sample đủ điều kiện. Outcome Accuracy có mẫu số riêng. Issue có thiếu hụt quyết định được loại khỏi điểm chính. Điểm của các criterion còn đủ căn cứ vẫn có trong chi tiết. `summary.json` báo hai metric cùng số mẫu và issue review theo category.

## Hướng dẫn chuẩn bị dữ liệu và đánh giá legacy

Các ví dụ lịch sử bên dưới gọi `run_application.sh` với schema bản án gốc và điểm LEGIT. Trong terminal chạy các ví dụ đó, đặt `export EVALUATION_PROFILE=legacy`.

Ví dụ xuyên suốt dùng checkpoint local:

```text
/workspace/storage-shared/models/Qwen3-4B
```

Repository runtime:

```text
/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
```

Bạn dùng môi trường `.venv` đã cài đầy đủ thư viện, đặt ở bất kỳ đâu. Các lệnh gọi `python` và `vllm` từ môi trường đã activate. Chạy các bước có inference trên B200.

Luồng chạy: statistic phân loại input gốc thành full/partial → host Qwen3-4B trong terminal 1 → lọc full giữ nguyên dữ liệu gốc → Application chỉ xử lý full → chạy thử/full → đọc kết quả. Judge là endpoint đã được cung cấp, không cần host lại Judge trong hướng dẫn này.

## 1. Activate môi trường hiện có

Mở hai terminal trên B200. Trong **mỗi terminal**, activate môi trường của bạn. Nếu đã activate thì bỏ qua bước này.

Hai dòng dưới có **một giá trị cần thay**: đường dẫn môi trường thực tế của bạn; đây không phải đường dẫn checkpoint hay repository.

```bash
export LEGAL_ENV_DIR="/duong/dan/thuc/te/den/moi-truong-venv-cua-ban"
source "$LEGAL_ENV_DIR/bin/activate"
```

Các bước sau giả định terminal đã dùng đúng môi trường này.

### 1.1. Terminal 2: phân loại dữ liệu trước khi chạy Application

Statistic chỉ cần Judge endpoint, không cần host Qwen. Chạy trên input gốc chưa có data_status; toàn bộ file nguồn được giữ nguyên. JSON/schema lỗi (ví dụ issues rỗng ở dòng 578) được ghi vào invalid_input_records.jsonl/errors.jsonl rồi bỏ qua; tiếp tục với các records hợp lệ. Statistic không gọi clean_input để sửa nguồn và không gán partial cho record lỗi. Nếu tất cả records đều lỗi hoặc file rỗng thì dừng trước API.

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
cd "$PROJECT_ROOT"
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export RAW_INPUT_JSONL="$PROJECT_ROOT/input/judgment_with_issues_sample.jsonl"
export STAT_RUN_ID="completeness-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

export JUDGE_BASE_URL=http://172.29.62.173:8000/v1
export JUDGE_MODEL=DeepSeek-V4.1-Flash-1017
export JUDGE_THINKING_MODE=deepseek_vllm_xhigh
export JUDGE_RESPONSE_FORMAT=json_schema
export JUDGE_API_KEY=EMPTY
export BATCH_SIZE=20
export JUDGE_CONCURRENCY=20
export MAX_OUTPUT_TOKENS=32768
export API_TIMEOUT_SECONDS=1800
export API_MAX_RETRIES=3
export JUDGE_JSON_RETRIES=2
export RETRY_BACKOFF_SECONDS=2

INPUT_JSONL="$RAW_INPUT_JSONL" RUN_ID="$STAT_RUN_ID" LIMIT=all RESUME=0 \
  bash scripts/statistic.sh

python -m json.tool "$OUTPUT_DIR/$STAT_RUN_ID/summary.json"
```

`full`: facts_clean đủ dữ kiện thực tế cho mọi issue đã được Tòa giải quyết; `partial`: ít nhất một issue thiếu tình tiết/chứng cứ quyết định trong application/conclusion. Diễn đạt tương đương hoặc thiếu diễn giải pháp luật không tự làm sample partial. Details có phân tích đủ issues và quotes được kiểm tra nguyên văn với đúng nguồn.

Nếu dùng classified_input.jsonl thì cần `classified_output_complete=true`. Nếu run còn failed, vẫn có thể dùng filter_full.sh ở mục 1.4 để xuất riêng các records đã được phân loại full thành công. Summary báo input/input_valid/input_invalid riêng; LIMIT được áp dụng lên records hợp lệ. `source_fully_classified=false` nếu còn dòng input lỗi bị bỏ qua hoặc chưa chọn hết nguồn. Các dòng lỗi không nằm trong output có nhãn. Nếu còn API/JSON/schema/evidence errors của Judge hoặc Judge không phân loại được, kết quả thành công được lưu vào `classified_input.partial.jsonl`, lỗi vào errors.jsonl; không có nhãn giả và không có classified_input.jsonl hoàn chỉnh. Tỷ lệ full/partial dùng số phân loại thành công làm mẫu số; tỷ lệ lỗi Judge báo riêng trên selected, input_invalid báo riêng trên tổng input. Resume bằng đúng exports và STAT_RUN_ID ở trên:

```bash
INPUT_JSONL="$RAW_INPUT_JSONL" RUN_ID="$STAT_RUN_ID" LIMIT=all RESUME=1 \
  bash scripts/statistic.sh
```

Các ví dụ resume/relocate này yêu cầu code còn khớp run gốc. Nếu đã cập nhật bản sửa retry này từ phiên bản cũ, dùng retry_statistic.sh ở mục 1.4 để nhập kết quả thành công vào run mới.

Nếu muốn tăng số samples đồng thời của **run statistic đang chạy**, nhấn Ctrl-C và đợi lệnh cũ thoát. Cập nhật hai file scripts/resume_statistic.sh và scripts/resume_statistic.py lên B200 rồi chạy trong môi trường .venv đã activate. Nếu project vẫn ở vị trí đã lưu trong config, chỉ cập nhật hai helper là đủ:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
cd "$PROJECT_ROOT"

# Thay TEN_RUN_DANG_CHAY bằng RUN_ID thật của run vừa dừng, không tạo ID mới.
bash scripts/resume_statistic.sh TEN_RUN_DANG_CHAY \
  --batch-size 200 --judge-concurrency 200
```

Nếu đang dùng OUTPUT_DIR khác mặc định, export lại OUTPUT_DIR đó trước lệnh trên. Helper tự đọc các giá trị gốc trong config.json, kể cả MAX_OUTPUT_TOKENS=65536 nếu run ban đầu dùng mức đó; chỉ đổi batch và Judge concurrency. Cấu hình cũ được backup, thay đổi được ghi vào resume_config_changes.jsonl. Các classification_details đã lưu được dùng lại, chỉ samples chưa thành công được gọi lại. Nếu run cũ chưa thoát, helper từ chối để không có hai process cùng ghi. Input, code, model, thinking và các thiết lập nội dung phải tiếp tục khớp run gốc. Tăng concurrency không thay đổi LIMIT hay nhãn đã có. Số request server thực sự chạy song song còn phụ thuộc scheduler/KV cache.

### 1.2. Resume statistic sau khi chuyển thư mục project

Nếu run nằm ở `/workspace/storage-shared/nlp/users/minhpn19/LegalApplication/outputs/20261117T135710_be74ac50` nhưng config.json còn lưu project_root cũ, không đọc path từ config đó để `cd`. Đồng bộ code đã sửa lên project mới, gồm scripts/resume_statistic.sh, scripts/resume_statistic.py và src/legal_application/config.py. Trong terminal đã activate .venv và sau khi process cũ thoát, chạy nguyên khối:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
cd "$PROJECT_ROOT"

bash scripts/resume_statistic.sh 20261117T135710_be74ac50 \
  --relocate \
  --batch-size 200 \
  --judge-concurrency 200
```

Helper kiểm tra input checksum, giữ nguyên RUN_ID/model/thinking/LIMIT và các thiết lập nội dung, sao lưu config thành config.before-resume-*.json, ghi resume_config_changes.jsonl rồi cập nhật project_root/output_dir. Các records đã phân loại thành công được bỏ qua, chỉ gọi Judge cho records chưa thành công. Input bên ngoài project vẫn giữ nguyên đường dẫn, cụ thể `/workspace/storage-shared/nlp/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl` trong trường hợp này. Input và classification_details không bị viết lại trong bước chuyển cấu hình. Log lịch sử được giữ nguyên để đối chiếu.

Nếu input cũng chuyển sang nơi khác, thêm option sau vào lệnh resume (thay bằng path thật):

```bash
bash scripts/resume_statistic.sh 20261117T135710_be74ac50 \
  --relocate --batch-size 200 --judge-concurrency 200 \
  --input-jsonl /duong/dan/moi/judgment_with_issues.jsonl
```

Input nằm trong project cũ được tự đổi sang vị trí tương ứng dưới project mới; --input-jsonl dùng khi muốn chỉ định riêng. Mọi thay đổi input path đều phải giữ nguyên bytes/checksum. Helper nhận diện chính xác phiên bản code chỉ khác giá trị PROJECT_ROOT mặc định để chuyển implementation hash tương ứng; thay đổi code/prompt/schema khác vẫn bị từ chối. Sau lần chuyển đầu, những lần resume tiếp theo không cần --relocate nếu paths giữ nguyên.

### 1.3. Kết quả statistic

Khi toàn bộ selected samples thành công, file classified_input.jsonl giữ nguyên toàn bộ object gốc, chỉ thêm data_status, theo đúng thứ tự input. LIMIT=all phân loại toàn bộ; nếu chỉ thử LIMIT=5 thì file hoàn chỉnh chỉ bao gồm 5 samples được chọn, không phải toàn dataset. Dùng RUN_ID mới nếu đổi LIMIT.

Ví dụ nguồn 2.010 dòng có một record issues rỗng tại dòng 578: counts.input=2010, input_valid=2009, input_invalid=1; với LIMIT=all có selected=2009. Khi 2.009 records này phân loại thành công, output có 2.009 records, classified_output_complete=true nhưng source_fully_classified=false. classification_details giữ input_index/input_line của nguồn gốc (có khoảng trống ở dòng lỗi); Application sau đó đánh số theo file classified_input.jsonl mới. Dùng doc_name để đối chiếu giữa hai giai đoạn.

Đọc log records nguồn bị bỏ qua, không cần sửa input:

```bash
python - "$OUTPUT_DIR/$STAT_RUN_ID/invalid_input_records.jsonl" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        print(row["input_line"], row["doc_name"], row["error"])
PY
```

Có thể dùng file có nhãn từ một run hoàn tất; Application tự bỏ qua partial:

```bash
export CLASSIFIED_INPUT_JSONL="$OUTPUT_DIR/$STAT_RUN_ID/classified_input.jsonl"
export INPUT_JSONL="$CLASSIFIED_INPUT_JSONL"
```

Cách xuất input giữ nguyên các trường gốc và tận dụng run còn failed được hướng dẫn ngay dưới đây. Chạy filter_full.sh rồi giữ manifest cạnh file raw đã lọc.

### 1.4. Xuất input_application_clean_v1 và chạy lại samples failed

Run `20261117T202530_5e87eebe` có 497 full, 313 partial và 1.199 failed. Có thể lấy ngay 497 full mà không đợi các samples failed thành công. Trong môi trường đã activate, sau khi cập nhật code mới lên B200:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export RAW_INPUT_JSONL=/workspace/storage-shared/nlp/users/minhpn19/LegalDataset/output/2k_clean_hoi_final/judgment_with_issues.jsonl
export CLASSIFICATION_DETAILS="$OUTPUT_DIR/20261117T202530_5e87eebe/classification_details.jsonl"
export FILTERED_JSONL="$PROJECT_ROOT/input/input_application_clean_v1.jsonl"
cd "$PROJECT_ROOT"

bash scripts/filter_full.sh \
  --input "$RAW_INPUT_JSONL" \
  --classifications "$CLASSIFICATION_DETAILS" \
  --output "$FILTERED_JSONL"

export CLASSIFIED_INPUT_JSONL="$FILTERED_JSONL"
export INPUT_JSONL="$FILTERED_JSONL"
```

Script nối bằng document_id nếu có ở cả hai file; code statistic cũ lưu doc_name nên có hỗ trợ khóa này. Khi cả hai khóa có mặt phải nhất quán. sample_sha256 trong artifact được so với record nguồn nếu có. Output giữ đúng thứ tự, bytes JSON, trường, giá trị và định dạng từng dòng nguồn; không thêm data_status, không bỏ trường metadata. Sample partial, failed hoặc chưa có nhãn không được xuất. Input gốc và classification_details không bị sửa. Dòng JSON/schema nguồn lỗi được ghi trong input_application_clean_v1.jsonl.filter_log.jsonl rồi bỏ qua; nếu một nhãn full không tìm được record tương ứng hoặc checksum sai, script dừng và giữ output cũ.

Giữ `input_application_clean_v1.jsonl.manifest.json` cạnh file output. File này lưu checksum và IDs của tập full để Application xác minh raw input không có nhãn. Nếu chuyển output sang nơi khác, copy cả JSONL và manifest cạnh nhau. Không dùng head/cắt/sửa file đã lọc; dùng LIMIT để chạy thử. Application không còn tùy chọn all/full_only và luôn chỉ xử lý full.

Để khôi phục 1.199 samples failed bằng code mới, tạo run mới và nhập 810 nhãn cũ đã thành công; không chỉnh implementation hash của config cũ và không gọi lại 810 samples đó. Input có thể đổi path nhưng bytes/checksum phải khớp run cũ:

```bash
export RETRY_RUN_ID="retry-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"
bash scripts/retry_statistic.sh 20261117T202530_5e87eebe \
  --run-id "$RETRY_RUN_ID" \
  --input-jsonl "$RAW_INPUT_JSONL" \
  --batch-size 200 \
  --judge-concurrency 200 \
  --judge-json-retries 4 \
  --max-output-tokens 65536
```

Helper khôi phục Judge/thinking/LIMIT/timeout và các thiết lập còn lại từ config cũ. --judge-json-retries 4 cho tối đa 5 lần structured-output; --max-output-tokens gửi đúng 65536, không tăng ngầm hoặc tự tắt thinking. Muốn thay thinking cho các request mới, thêm --judge-thinking-mode high/off/... như bảng thinking phía dưới. Metadata từng kết quả giữ thông tin request thực tế, kể cả kết quả đã nhập từ run cũ. Các nhãn cũ được kiểm tra lại identity, sample hash, Judge model, schema, raw response và evidence trước mọi request mới.

Nếu ngắt run khôi phục, giữ RETRY_RUN_ID đã in ở đầu log và resume bình thường với code hiện tại:

```bash
bash scripts/resume_statistic.sh "$RETRY_RUN_ID" \
  --batch-size 200 --judge-concurrency 200
```

Run mới có classification_details gồm cả kết quả nhập và kết quả mới. Sau khi chạy xong (kể cả còn failed), lọc lại để cập nhật tập full:

```bash
bash scripts/filter_full.sh \
  --input "$RAW_INPUT_JSONL" \
  --classifications "$OUTPUT_DIR/$RETRY_RUN_ID/classification_details.jsonl" \
  --output "$FILTERED_JSONL"

export CLASSIFIED_INPUT_JSONL="$FILTERED_JSONL"
export INPUT_JSONL="$FILTERED_JSONL"
```

Chỉ cập nhật input_application_clean_v1 trước khi tạo Application run mới; đổi file sau khi đã chạy Application sẽ làm resume bị từ chối đúng theo checksum. Với nhiều run độc lập, có thể lặp --classifications; nhãn xung đột bị từ chối.

Thông báo `classification exhausted structured-output retries` chưa xác định nguyên nhân. Helper ghi thống kê validation_errors cuối của các samples còn failed trong `$OUTPUT_DIR/.statistic-seeds/$RETRY_RUN_ID/source.json` và in ra khi bắt đầu. Đọc nguyên nhân của run cũ bằng lệnh chỉ đọc sau (không gọi API):

```bash
python - "$OUTPUT_DIR/20261117T202530_5e87eebe/errors.jsonl" <<'DIAG'
import collections
import json
import sys
latest = {}
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        error = json.loads(line)
        if error.get("stage") == "classification":
            latest[error["doc_name"]] = error
reasons = collections.Counter()
finishes = collections.Counter()
for error in latest.values():
    metadata = error.get("metadata") or {}
    reasons[(metadata.get("validation_errors") or [error["error"]])[-1]] += 1
    calls = metadata.get("judge_calls") or []
    if calls:
        finishes[str(calls[-1].get("finish_reason"))] += 1
print("Last validation errors:")
for reason, count in reasons.most_common():
    print(count, reason)
print("Final finish_reason:", dict(finishes))
DIAG
```

Record lỗi đã gửi có finish_reason=stop cả ba lần; validation_errors lần lượt báo evidence quote không nằm trong facts_clean/application. Record này bị loại vì trích dẫn không khớp nguồn, không phải vì response bị cắt. Đây là nguyên nhân đã xác nhận cho record đó; dùng thống kê để kiểm tra các records còn lại.

Bản sửa dùng evidence_format=source_ids_v1: gửi toàn bộ benchmark_seed cùng evidence_catalog gồm các đoạn gốc liên tục được đánh ID, ví dụ facts_clean:000001 hoặc application:000001. Mỗi đoạn tối đa khoảng 500 ký tự, nhưng không giới hạn tổng số đoạn và không bỏ phần nguồn nào. Judge đọc các đoạn theo ý nghĩa rồi trả ID thay vì tự chép lại quote. reference_evidence dùng {source, quote_id}; facts_clean_evidence/raw_facts_evidence chứa danh sách ID thuộc đúng nguồn. Schema giới hạn ID hợp lệ; Python kiểm tra ID/nguồn, lấy nguyên văn đoạn gốc, rồi chạy lại validator nhãn, mapping và evidence hiện có. Không đổi full/partial hoặc số tiền/câu chữ để vượt kiểm tra; ID sai hoặc dùng nhầm nguồn vẫn bị loại.

classification_details giữ cấu trúc evidence cũ với quote nguyên văn và các đoạn facts_clean/facts đã mở rộng; raw_judge_response giữ đúng JSON IDs của model. metadata.evidence_format ghi giao thức; metadata.normalizations ghi ID, nguồn và start/end của từng đoạn được mở rộng. Các kết quả quote cũ vẫn nhập/resume được nếu hợp lệ; chỉ căn chỉnh khoảng trắng về đúng nguồn, không fuzzy matching. Retry gửi JSON đã parse được kèm vị trí field lỗi. Lỗi cuối có last_validation_error, invalid_responses (preview final content), usage/finish_reason/requested_max_tokens; summary có failure_reasons. Raw reasoning không được lưu.

Sau khi cập nhật bản sửa giao thức evidence, dùng retry_statistic.sh với run nguồn cần giữ kết quả như lệnh ở trên để tạo RUN_ID mới. Nếu đã chạy một run retry bằng code trước đó, có thể dùng chính RUN_ID retry đó làm source_run; helper sẽ nhập toàn bộ kết quả thành công hiện có, kể cả kết quả đã nhập từ run đầu. Không resume trực tiếp qua thay đổi code/hash.

Nếu phần lớn lỗi là finish_reason=length hoặc final content rỗng, cần xem token usage và cấu hình reasoning parser/context của server; prompt/retry phía client không bảo đảm khắc phục hết giới hạn server. [Tài liệu vLLM về reasoning](https://docs.vllm.ai/en/stable/features/reasoning_outputs/) và [structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/) mô tả việc tách reasoning/final và hỗ trợ theo backend. Nếu còn lỗi schema/evidence thì response vẫn bị từ chối, không chuyển thành full để tăng số dữ liệu giả.

### 1.5. Tạo ba input độc lập cho issue spotting, application và conclusion

Nguồn là input/input_application_clean_v2.jsonl gồm 1.207 records full do bạn tạo. Script chuyển dữ liệu không gọi API, không cần GPU, giữ nguyên nguồn và không đổi code/hash của package pipeline. Chạy trong môi trường Python đã activate:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
cd "$PROJECT_ROOT"

bash scripts/create_task_inputs.sh \
  --input "$PROJECT_ROOT/input/input_application_clean_v2.jsonl" \
  --output-dir "$PROJECT_ROOT/input/input_temp_18_11" \
  --expected-count 1207
```

Ba output có cùng số dòng, cùng thứ tự nguồn, mỗi dòng chỉ có input và expected_answer:

| File trong input/input_temp_18_11 | input | expected_answer |
|---|---|---|
| input_issue_spotting.jsonl | Chuỗi benchmark_seed.facts_clean nguyên văn | Danh sách issues nguyên gốc |
| input_application.jsonl | Object gồm facts_clean, issues, rules | Chuỗi benchmark_seed.application nguyên văn |
| input_conclusion.jsonl | Object gồm facts_clean, issues, rules, application | Chuỗi benchmark_seed.conclusion nguyên văn |

Facts của cả ba tác vụ đều dùng facts_clean. Trường application trong input_conclusion là benchmark_seed.application, không phải output của model. Đây là các bộ dữ liệu độc lập; kết quả issue spotting/application không được ghép tự động sang tác vụ sau.

Ví dụ cấu trúc một dòng Application (giá trị chỉ minh họa):

```json
{
  "input": {
    "facts_clean": "Các tình tiết đã được làm sạch.",
    "issues": ["Vấn đề pháp lý thứ nhất?"],
    "rules": [
      {
        "law_str": "Tên văn bản luật",
        "citation_str": "Điều 1",
        "context_text": "Nguyên văn nội dung điều khoản."
      }
    ]
  },
  "expected_answer": "Nội dung benchmark_seed.application."
}
```

rules được trải phẳng theo thứ tự gốc: mỗi provision là một object gồm law_str, citation_str và context_text. Nội dung lấy từ laws[].provisions[].law_content.context_text; không đưa các metadata luật khác vào input. Khi luật không có provisions, lấy laws[].law_content.context_text, hoặc exact_text nếu context_text thiếu/null; citation_str lúc này là null. Quy tắc khớp run_application, không cắt ngắn, lọc bớt hoặc viết lại nội dung luật.

manifest.json trong cùng thư mục lưu checksum nguồn/output, số records và ánh xạ số dòng sang doc_name/document_id. Nó giúp đối chiếu ba file mà không thêm field vào từng sample. --expected-count chỉ kiểm tra số lượng, không giới hạn dữ liệu. Không có hidden cap; muốn dùng nguồn khác thì đổi --input và --expected-count tương ứng, hoặc bỏ --expected-count. TASK_SOURCE_JSONL/TASK_INPUT_DIR là hai biến tùy chọn cho script; không kế thừa INPUT_JSONL/OUTPUT_DIR của các run benchmark trước.

Script dừng khi gặp JSON/schema lỗi, doc_name trùng, nhãn data_status có mặt nhưng khác full, hoặc số samples khác expected-count. Nó không tự bỏ dòng và không ghi đè ba output cũ khi validation thất bại; file nguồn không bị sửa. Nguồn full từ filter_full không có data_status vẫn được chấp nhận. Chạy lại thành công sẽ thay ba output và manifest bằng dữ liệu mới.

Các file hai trường này phục vụ từng tác vụ riêng. run_application hiện tại đọc schema bản án gốc, nên vẫn dùng input_application_clean_v2.jsonl khi chạy pipeline benchmark hiện có.

## 2. Terminal 1: host checkpoint Qwen3-4B

Ví dụ dùng GPU số 0, port 8001, tên model trên API là `Qwen/Qwen3-4B`.

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
export MODEL_PATH=/workspace/storage-shared/models/Qwen3-4B
export MODEL_NAME=Qwen/Qwen3-4B

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/outputs/server_logs"

export HOST_LOG="$PROJECT_ROOT/outputs/server_logs/qwen3-4b-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}.log"

set -o pipefail
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 20 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
  --max-model-len 131072 \
  2>&1 | tee "$HOST_LOG"
```

Giữ terminal này chạy. Sau khi server khởi động, chuyển sang terminal 2.

Ý nghĩa các giá trị cần biết:

| Giá trị | Ý nghĩa |
|---|---|
| `MODEL_PATH` | Thư mục checkpoint đã có trên B200; vLLM đọc weights/tokenizer từ đây |
| `--served-model-name` | Tên nhận diện trên API; client phải gửi đúng tên này |
| `CUDA_VISIBLE_DEVICES=0` | Chọn GPU 0; sửa số GPU nếu muốn dùng GPU khác |
| `--port 8001` | Endpoint generator sẽ là `http://127.0.0.1:8001/v1` khi client cùng máy |
| `--max-num-seqs 20` | Giới hạn sequences của scheduler phía server; client cũng cấu hình concurrency 20 ở bước 3 |
| `--reasoning-parser qwen3` | Tách final answer khỏi reasoning của Qwen3 |
| `--max-model-len 131072` | Giới hạn context gồm cả input và output, với YaRN được cấu hình ở dòng trước |

Input có thể chứa toàn văn luật dài nên ví dụ dùng context 131072 và output budget 32768. YaRN là cấu hình mở rộng context, có thể ảnh hưởng hành vi model; khi so sánh các runs, giữ nhất quán cấu hình host và lưu lệnh host cùng kết quả. Không chỉ tăng `--max-model-len` mà bỏ cấu hình YaRN cho checkpoint Qwen3 này. Cách mở rộng context và parser tham khảo [Qwen: vLLM](https://qwen.readthedocs.io/en/stable/deployment/vllm.html); cách truyền config qua `--hf-overrides` tham khảo [vLLM CLI](https://docs.vllm.ai/en/latest/cli/serve/#--hf-overrides).

## 3. Terminal 2: cấu hình pipeline

Sau khi activate môi trường ở terminal 2, copy toàn bộ khối sau:

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
cd "$PROJECT_ROOT"

# Input full đã lọc ở bước 1.4, giữ file manifest bên cạnh; hoặc dùng classified_input có nhãn.
export INPUT_JSONL="$CLASSIFIED_INPUT_JSONL"

# Model được đánh giá, đã host ở terminal 1.
export GENERATOR_BASE_URL=http://127.0.0.1:8001/v1
export GENERATOR_MODEL=Qwen/Qwen3-4B
export GENERATOR_API_KEY=EMPTY
export GENERATOR_THINKING_MODE=none

# Judge chấm bài làm; endpoint này đã được cung cấp.
export JUDGE_BASE_URL=http://172.29.62.173:8000/v1
export JUDGE_MODEL=DeepSeek-V4.1-Flash-1017
export JUDGE_API_KEY=EMPTY
export JUDGE_THINKING_MODE=deepseek_vllm_xhigh
export JUDGE_RESPONSE_FORMAT=json_schema

# Mỗi batch tối đa 20 samples, mỗi giai đoạn tối đa 20 requests đồng thời.
export BATCH_SIZE=20
export GENERATOR_CONCURRENCY=20
export JUDGE_CONCURRENCY=20

# Giới hạn output cho MỖI request, gồm cả generator và Judge.
export MAX_OUTPUT_TOKENS=32768

export API_TIMEOUT_SECONDS=1800
export API_MAX_RETRIES=3
export JUDGE_JSON_RETRIES=2
export RETRY_BACKOFF_SECONDS=2

export OUTPUT_DIR="$PROJECT_ROOT/outputs"
export MODE=full
export LIMIT=all
export RESUME=0
export GENERATION_JSONL=""
export RUN_ID=""
```

`GENERATOR_THINKING_MODE=none` có nghĩa client không gửi tham số reasoning riêng cho generator. Qwen3 dùng mặc định thinking của server/template. `JUDGE_THINKING_MODE=deepseek_vllm_xhigh` mặc định gửi `chat_template_kwargs={"thinking":true,"reasoning_effort":75}`. Cả hai biến hỗ trợ đầy đủ bảng lựa chọn sau; đổi thinking mode cần RUN_ID mới.

| Giá trị | Ý nghĩa |
|---|---|
| `low`, `high`, `xhigh`, `max` | DeepSeek V4.1 vLLM: bật thinking với effort lần lượt 25, 50, 75, 100 |
| Số nguyên `1`–`100` | DeepSeek V4.1 vLLM: bật thinking với effort cụ thể, ví dụ 63 |
| `on`, `off` | DeepSeek: bật/tắt thinking; không gửi effort |
| `default`, `none` | Không gửi tham số thinking; none giữ nghĩa cũ của pipeline, KHÔNG phải tắt |
| `qwen3_on`, `qwen3_off` | Bật/tắt thinking cho Qwen3 bằng enable_thinking |
| `deepseek_vllm_<mức>` | Alias tương đương: deepseek_vllm_low/high/xhigh/max/on/off hoặc deepseek_vllm_63 |

Không dùng các mức effort DeepSeek cho Qwen3. `medium` và `minimal` không phải tier của DeepSeek V4.1 vLLM, nên bị báo lỗi thay vì tự đoán mapping. Đây là interface của model self-hosted, khác [DeepSeek hosted API](https://api-docs.deepseek.com/api/create-chat-completion/). Bảng 25/50/75/100 và range 1–100 được đối chiếu với [recipe vLLM](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash#reasoning-and-tool-calling) và adapter LegalDataset. Effort không phải giới hạn token; MAX_OUTPUT_TOKENS vẫn giữ vai trò riêng.

Ví dụ đổi Judge sang max, hoặc chọn budget cụ thể, trong một run mới:

```bash
export JUDGE_THINKING_MODE=max

# Thay dòng trên bằng dòng này nếu muốn effort 63:
# export JUDGE_THINKING_MODE=63

# Nếu generator là Qwen3 và muốn tắt thinking của nó:
# export GENERATOR_THINKING_MODE=qwen3_off
```

config.json/run.log ghi mode yêu cầu và tham số thực tế cho từng vai trò; request metadata ghi actual_thinking_parameters. Giữ xhigh nếu muốn dùng cấu hình Judge mặc định của benchmark.

Nếu dùng cùng endpoint DeepSeek cho cả hai vai trò, export các giá trị sau trước khi tạo run mới:

```bash
export GENERATOR_BASE_URL="$JUDGE_BASE_URL"
export GENERATOR_MODEL="$JUDGE_MODEL"
export GENERATOR_API_KEY="$JUDGE_API_KEY"
export GENERATOR_THINKING_MODE=deepseek_vllm_xhigh
export JUDGE_THINKING_MODE=deepseek_vllm_xhigh
```

Nếu model host ở **máy khác**, chỉ đổi `GENERATOR_BASE_URL` thành IP/hostname thật của máy host, giữ `/v1` ở cuối. Ví dụ cú pháp:

```bash
# Thay HOST_THUC_TE bằng IP/hostname máy host model trước khi dùng.
export GENERATOR_BASE_URL="http://HOST_THUC_TE:8001/v1"
```

Nếu vẫn chạy cùng máy như bước 2 thì giữ `http://127.0.0.1:8001/v1`, không cần chạy ví dụ đổi host.

## 4. Kiểm tra endpoint đã sẵn sàng

Hai lệnh sau chỉ đọc danh sách models, không sinh bài làm benchmark:

```bash
curl --fail --silent --show-error \
  -H "Authorization: Bearer $GENERATOR_API_KEY" \
  "$GENERATOR_BASE_URL/models"

curl --fail --silent --show-error \
  -H "Authorization: Bearer $JUDGE_API_KEY" \
  "$JUDGE_BASE_URL/models"
```

Ở endpoint generator phải có model mang tên `Qwen/Qwen3-4B`. Nếu server còn đang load, đợi terminal 1 báo sẵn sàng rồi gọi lại. Nếu endpoint thực tế yêu cầu API key, export key được cấp vào biến của vai trò tương ứng trước khi chạy.

## 5. Làm sạch và validate toàn bộ input

Chạy bước này trước lần benchmark đầu, hoặc sau khi đổi sang file input khác:

```bash
export CHECK_RUN_ID="qwen3-4b-check-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=validate-only LIMIT=all RUN_ID="$CHECK_RUN_ID" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Đọc báo cáo:

```bash
python -m json.tool "$OUTPUT_DIR/$CHECK_RUN_ID/summary.json"
python -m json.tool "$OUTPUT_DIR/$CHECK_RUN_ID/input_cleanup.json"
```

Hành vi của bước này:

- Quét toàn bộ input, kể cả ngoài LIMIT. Input raw từ filter_full.sh được kiểm tra bằng manifest/checksum; input khác cần nhãn data_status hợp lệ. Application chỉ chọn full, không đoán nhãn cho dữ liệu chưa phân loại.
- Ghi dòng không parse được JSON/UTF-8 vào `removed_input_lines.jsonl`, rồi **xóa dòng đó khỏi chính INPUT_JSONL** và tiếp tục với dòng sau.
- Giữ nguyên nội dung/thứ tự các dòng còn lại; LIMIT tính trên samples hợp lệ sau cleanup.
- Nếu JSON parse được nhưng thiếu trường bắt buộc, issues rỗng/sai kiểu hoặc thiếu nội dung luật, lưu nguyên record vào removal log rồi xóa khỏi input và tiếp tục. Với doc_name trùng, giữ record hợp lệ đầu tiên, log và loại các bản sau. Không tự sinh issues/facts/luật để bù dữ liệu thiếu.
- Không gọi generator hoặc Judge.

Tệp mẫu raw trong workspace có 6 records hợp lệ nhưng chưa gắn nhãn; phải chạy statistic ở bước 1.1 trước. Statistic không tự xóa lỗi nguồn. Với file có 6 records thì chỉ chạy được tối đa 6 samples; cần input có ít nhất 20 records hợp lệ đủ điều kiện filter để có batch đủ 20 samples.

Đọc chi tiết các dòng đã xóa mà không in bản base64 dài:

```bash
python - "$OUTPUT_DIR/$CHECK_RUN_ID/removed_input_lines.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        print(f"Input: {row['input_path']}")
        print(f"Dòng gốc: {row['original_input_line']}")
        print(f"Lỗi: {row['error_type']}: {row['error']}")
        print(f"Độ dài bytes: {row['byte_length']}")
        print("Nguyên văn:")
        print(row["raw_line"] if row["raw_line"] is not None else "UTF-8 không hợp lệ; bytes gốc nằm trong raw_line_base64")
        print()
PY
```

Nếu cleanup xảy ra trong run validate này, log nằm trong thư mục `$CHECK_RUN_ID`. Những run sau đọc input sạch nên removal log của các run sau có thể rỗng. Log gốc vẫn được giữ trong run đã thực hiện xóa.

## 6. Chạy thử 5 samples, gồm generation và chấm điểm

Sau khi bước 5 thành công:

```bash
export SMOKE_RUN_ID="qwen3-4b-smoke-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=full LIMIT=5 RUN_ID="$SMOKE_RUN_ID" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Đọc kết quả:

```bash
python -m json.tool "$OUTPUT_DIR/$SMOKE_RUN_ID/summary.json"
```

Nếu chỉ muốn thử 1 sample thì dùng **RUN_ID mới**:

```bash
export SMOKE_ONE_RUN_ID="qwen3-4b-smoke1-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=full LIMIT=1 RUN_ID="$SMOKE_ONE_RUN_ID" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Không cần tạo file con cho smoke test. Dù LIMIT=5, pipeline vẫn kiểm tra/làm sạch toàn bộ INPUT_JSONL trước khi chọn 5 samples.

## 7. Chạy toàn bộ dataset, concurrency 20

Giữ INPUT_JSONL đang dùng nếu muốn chạy toàn bộ file đó. Nếu chuyển sang dataset lớn, export đường dẫn **file thực tế đã có** vào INPUT_JSONL trước khối lệnh dưới; không bắt buộc file phải tên `full_dataset.jsonl`.

```bash
# Chỉ bỏ comment và thay tên file nếu bạn chuyển sang input khác:
# export INPUT_JSONL="$PROJECT_ROOT/outputs/TEN_RUN_PHAN_LOAI/classified_input.jsonl"

export FULL_RUN_ID="qwen3-4b-full-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=validate-only LIMIT=all RUN_ID="${FULL_RUN_ID}-validate" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh" \
  && MODE=full LIMIT=all RUN_ID="$FULL_RUN_ID" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Lệnh full chỉ chạy khi validation thành công. Cấu hình đã export ở bước 3 sẽ xử lý mỗi batch tối đa 20 samples: tối đa 20 generator requests đồng thời, sau đó tối đa 20 Judge requests đồng thời. Batch cuối ít hơn 20 vẫn chạy hết. Không có giới hạn tổng mặc định 1000 samples.

Nếu muốn **tổng cộng 20 samples**, dùng LIMIT=20 trong một run mới. Nếu muốn **toàn bộ dataset nhưng mỗi lượt 20**, dùng LIMIT=all như trên.

Application luôn chỉ chọn full trước LIMIT trong cả full, judge-only và validate-only; không còn tùy chọn DATA_STATUS_USED. Ví dụ chọn tối đa 100 samples full:

```bash
MODE=full LIMIT=100 RESUME=0 \
  RUN_ID="qwen3-full-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}" \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Nếu đọc file có nhãn, partial vẫn ở nguyên nguồn nhưng không được gọi model. Nếu đọc input_application_clean_v1.jsonl không có nhãn, manifest xác minh rằng toàn bộ file đã được lọc full. Summary có data_status_counts.input/selected/completed, filtered_out và eligible. Không đổi nội dung input/manifest giữa các lần resume.

Giới hạn concurrency là số request client có thể gửi đồng thời; số sequences server thực sự chạy tại một thời điểm còn phụ thuộc scheduler và KV cache.

## 8. Theo dõi tiến trình và đọc artifacts

Khi chạy trực tiếp trong terminal, tqdm tự hiện thanh **Scan input**, **Validate input**, **Generation** và **Judge**. Hai thanh đầu theo bytes; hai thanh sau theo samples của toàn run, hiển thị batch, thời gian, tốc độ và ETA. Mỗi sample được cập nhật ngay khi xử lý xong, kể cả khi các requests trả về khác thứ tự.

Các số phía sau thanh: `ok` = kết quả đã lưu hợp lệ, `failed` = lỗi trong invocation hiện tại, `skipped` = kết quả đã có đầu invocation, `review` = evaluation cần xem xét, `blocked` = không gọi Judge vì thiếu generation. Thanh đạt 100% là đã xử lý hết samples; đọc các số và summary để biết thành công/thất bại. Khi resume, số kết quả đã lưu được đưa vào initial progress, không gọi lại hoặc cộng trùng. Judge-only hiển thị Import generations và Judge.

Progress được ghi ra stderr; console logging dùng tqdm.write để không ghi đè thanh. run.log vẫn là log văn bản chi tiết, không chứa mã điều khiển của thanh. Khi redirect stderr ra file hoặc môi trường không có TTY, thanh tự ẩn; theo dõi run.log bằng lệnh dưới.

Trong terminal mới, chỉ cần khai báo root, output và tên run đang theo dõi. `FULL_RUN_ID` bên dưới phải là **tên chính xác đã in/sử dụng ở bước 7**, không tạo timestamp mới.

Nếu vẫn ở terminal 2 sau khi full run kết thúc, biến FULL_RUN_ID đã có sẵn:

```bash
export RESULT_DIR="$OUTPUT_DIR/$FULL_RUN_ID"

python -m json.tool "$RESULT_DIR/summary.json"
wc -l "$RESULT_DIR/generations.jsonl" "$RESULT_DIR/evaluations.jsonl" "$RESULT_DIR/errors.jsonl"
```

Để theo dõi khi run còn đang chạy, mở terminal khác, đặt RESULT_DIR thành thư mục run thật và chạy:

```bash
# Thay TEN_RUN_THUC_TE bằng tên run bạn muốn theo dõi.
export RESULT_DIR=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication/outputs/TEN_RUN_THUC_TE
tail -n 100 -f "$RESULT_DIR/run.log"
```

Ctrl-C ở lệnh `tail` chỉ dừng việc xem log. Các file trong một run:

| File | Nội dung |
|---|---|
| `generations.jsonl` | Một bài làm raw cho mỗi bản án generation thành công |
| `evaluations.jsonl` | Nhãn Judge theo từng issue, phán quyết model/tham chiếu, lý do và điểm Python tính |
| `summary.json` | Input, số dòng xóa, selected/completed/failed/review, mean score và phân bố điểm |
| `run.log` | Tiến trình từng batch, request, retry và lỗi |
| `config.json` | Snapshot cấu hình để đối chiếu/resume, không lưu API keys |
| `errors.jsonl` | Lỗi generation/Judge hoặc thiếu generation source theo sample |
| `removed_input_lines.jsonl` | Dòng input bị xóa, số dòng gốc, lý do, raw text/base64 và checksum |
| `input_cleanup.json` | Hash trước/sau và các transaction làm sạch input |

Để xem **một generation và một evaluation**:

```bash
python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ["generations.jsonl", "evaluations.jsonl"]:
    with (root / name).open(encoding="utf-8") as handle:
        first = next(handle, None)
    print(f"\n===== {name}: record đầu tiên =====")
    print(json.dumps(json.loads(first), ensure_ascii=False, indent=2) if first else "Chưa có record")
PY
```

Output concurrent có thể khác thứ tự dòng input. Dùng `doc_name` + `input_index` và checksum để join; không ghép hai JSONL theo vị trí dòng.

## 9. Resume một run bị gián đoạn hoặc còn sample lỗi

**Trong cùng terminal, cùng các exports của run full ở bước 7**, chạy:

```bash
MODE=full LIMIT=all RUN_ID="$FULL_RUN_ID" RESUME=1 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Resume smoke 5 samples tương tự nhưng phải giữ LIMIT=5:

```bash
MODE=full LIMIT=5 RUN_ID="$SMOKE_RUN_ID" RESUME=1 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Generation đã lưu thành công không bị gọi lại. Nếu chỉ Judge lỗi, resume chỉ chấm lại sample đó. Evaluation `completed` hoặc `needs_review` đã lưu đều được bỏ qua. Muốn chấm lại needs_review, tạo run judge-only mới theo bước 10.

Resume yêu cầu giữ nguyên model, endpoints, input hợp lệ, LIMIT, batch/concurrency, output budget, timeout/retry và code/prompt/schema. Đổi các giá trị này thì dùng RUN_ID mới. Nếu có JSON lỗi mới trong input nhưng xóa chúng đưa file về đúng hash của input sạch cũ, resume vẫn tiếp tục được. Việc thay đổi valid records/thứ tự của chúng làm resume bị từ chối trước khi xóa input.

### Khôi phục cấu hình nếu đã đóng terminal cũ

Khối dưới dành cho Application khi project vẫn ở vị trí đã lưu trong config. Với statistic, dùng scripts/resume_statistic.sh ở mục 1.1; nếu vừa di chuyển project, dùng --relocate ở mục 1.2 trước khi khôi phục biến từ config cũ. Activate môi trường hiện có, đặt đường dẫn chính xác tới config.json của run cần resume, rồi copy khối dưới. Python chỉ chuyển các giá trị cấu hình đã lưu thành shell exports, không gọi API hoặc kiểm tra phiên bản thư viện.

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/users/minhpn19/LegalApplication
cd "$PROJECT_ROOT"

# Thay TEN_RUN_THUC_TE bằng tên run cần tiếp tục.
export RESUME_CONFIG="$PROJECT_ROOT/outputs/TEN_RUN_THUC_TE/config.json"

RESTORE_FILE="$(mktemp)"
if python - "$RESUME_CONFIG" > "$RESTORE_FILE" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)

fields = [
    "project_root", "input_jsonl", "output_dir", "run_id", "mode", "generation_jsonl", "limit",
    "generator_base_url", "generator_model", "judge_base_url", "judge_model",
    "batch_size", "generator_concurrency", "judge_concurrency", "max_output_tokens",
    "timeout_seconds", "api_max_retries", "judge_json_retries", "retry_backoff",
    "generator_thinking_mode", "judge_thinking_mode", "judge_response_format",
]
aliases = {"timeout_seconds": "API_TIMEOUT_SECONDS", "retry_backoff": "RETRY_BACKOFF_SECONDS"}
exports = []
for field in fields:
    value = config[field]
    if value is None:
        value = "all" if field == "limit" else ""
    name = aliases.get(field, field.upper())
    exports.append(f"export {name}={shlex.quote(str(value))}")
print("\n".join(exports))
PY
then
  source "$RESTORE_FILE"
  rm -f "$RESTORE_FILE"

  # Nếu endpoint có authentication, export lại API keys gốc của bạn tại đây.
  RESUME=1 bash "$PROJECT_ROOT/scripts/run_application.sh"
else
  rm -f "$RESTORE_FILE"
  echo "Không đọc được config; chưa chạy resume." >&2
fi
```

Nếu resume full và còn generation chưa xong, server ở terminal 1 phải đang chạy với cùng checkpoint/cấu hình host. Resume judge-only chỉ cần Judge endpoint và generation source đã lưu.

## 10. Chạy judge-only từ generation đã lưu

Ví dụ chấm lại **run full** đã chạy ở bước 7, dùng cùng INPUT_JSONL và LIMIT=all. Nếu dùng generation của smoke thì thay nguồn và LIMIT tương ứng, ví dụ LIMIT=5.

```bash
export SOURCE_RUN_ID="$FULL_RUN_ID"
export JUDGE_RUN_ID="qwen3-4b-judge-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=judge-only LIMIT=all RUN_ID="$JUDGE_RUN_ID" RESUME=0 \
  GENERATION_JSONL="$OUTPUT_DIR/$SOURCE_RUN_ID/generations.jsonl" \
  bash "$PROJECT_ROOT/scripts/run_application.sh"

python -m json.tool "$OUTPUT_DIR/$JUDGE_RUN_ID/summary.json"
```

Nếu đang ở terminal mới, cần chạy lại phần exports ở bước 3, đặt SOURCE_RUN_ID thành tên run thật và INPUT_JSONL thành đúng input của run nguồn. Judge-only không gọi generator và không cần giữ Qwen server hoạt động. Model của bài làm được lấy từ generation record. Một selected sample không có generation sẽ được báo lỗi riêng; các sample có generation vẫn được chấm.

## 11. Ba mode và các loại lỗi

| Mode | Hành vi |
|---|---|
| `full` | Làm sạch/validate input → generation → Judge → tính điểm |
| `judge-only` | Làm sạch/validate input → đọc generation source → Judge → tính điểm |
| `validate-only` | Làm sạch/validate input, lưu báo cáo; không gọi model |

| Tình huống | Pipeline xử lý |
|---|---|
| JSON input hỏng, như chuỗi chưa đóng | Log đầy đủ rồi xóa dòng khỏi INPUT_JSONL, tiếp tục đọc samples sau |
| JSON parse được nhưng thiếu facts_clean/issues/luật, issues rỗng/sai kiểu | Log với validation_stage=schema, xóa record khỏi input và tiếp tục các samples hợp lệ |
| doc_name trùng | Giữ record hợp lệ đầu tiên, log và loại các bản sau với validation_stage=duplicate_doc_name |
| Toàn input không còn record sau cleanup | Giữ removal log và báo lỗi input rỗng |
| Timeout/connection lỗi, HTTP 408/409/429/5xx | Retry theo API_MAX_RETRIES và backoff; hết lượt thì ghi failed sample, tiếp tục các sample khác |
| Generator không giải quyết đủ issues nhưng có final text hợp lệ | Lưu nguyên bài làm để Judge chấm coverage/correctness; không tự sinh lại để sửa bài |
| Generator response bị cắt (`finish_reason=length`) | Không coi là generation thành công; ghi lỗi sample, tiếp tục các sample khác |
| Judge trả JSON/schema/mapping lỗi hoặc bị cắt | Regenerate JSON trong JUDGE_JSON_RETRIES; hết lượt thì ghi failed sample, tiếp tục các sample khác |
| Judge không suy ra được gold đáng tin cậy | Gắn needs_review, điểm null, không đưa vào trung bình hợp lệ |
| Generation source của judge-only hỏng JSON/sai mapping | Dừng preflight; không tự xóa generation source |
| RUN_ID đã tồn tại và RESUME=0 | Dừng, không ghi đè run cũ |
| Resume khác input hợp lệ/config/code | Từ chối resume; cần RUN_ID mới |

Các request HTTP 400/401/404/422 không được tự retry bằng cách đổi tham số. Với Judge HTTP 400 do endpoint không nhận `json_schema`, chỉ đổi sang JSON object khi endpoint thực tế hỗ trợ cách đó, trong **run mới**:

```bash
export JUDGE_RESPONSE_FORMAT=json_object
```

Client vẫn validate cùng schema và tính điểm bằng Python. Thinking mode được giữ theo JUDGE_THINKING_MODE đã export.

Nếu gặp lỗi context, input cộng output đã vượt giới hạn server. Pipeline ghi lỗi, không tự cắt luật/facts hoặc giảm MAX_OUTPUT_TOKENS. Giá trị 65536 cho MAX_OUTPUT_TOKENS chỉ dùng khi **cả hai endpoints** hỗ trợ budget/context tương ứng; thay giá trị đó cần run mới.

Exit code: `0` = hoàn tất/validate thành công (có thể còn needs_review); `1` = run còn sample failed; `2` = lỗi cấu hình/preflight/mapping; `130` = Ctrl-C. Xem summary và run.log để phân biệt nguyên nhân.

## 12. Giữ tiến trình khi chạy qua SSH

Nếu bạn dùng tmux, có thể mở session riêng cho server và benchmark:

```bash
# Ở terminal host, tạo session rồi activate env và chạy bước 2 bên trong.
tmux new -s legal-qwen-host
```

```bash
# Ở terminal benchmark, tạo session rồi activate env và chạy bước 3 trở đi bên trong.
tmux new -s legal-application
```

Nhấn Ctrl-B rồi D để detach, tiến trình vẫn chạy. Gắn lại session bằng:

```bash
tmux attach -t legal-qwen-host
```

```bash
tmux attach -t legal-application
```

Để dừng có kiểm soát, nhấn Ctrl-C trong terminal đang chạy benchmark; sau đó dùng RESUME=1 với đúng cấu hình để tiếp tục. Khi không còn cần generation, Ctrl-C trong terminal server để dừng vLLM.

## 13. Các file và biến thường cần đổi

| Nhu cầu | Chỗ cần đổi |
|---|---|
| Đổi checkpoint local | MODEL_PATH ở lệnh host bước 2 |
| Đổi tên model trên API | `--served-model-name` ở bước 2 và GENERATOR_MODEL ở bước 3, phải khớp nhau |
| Đổi IP/port generator | `--host`/`--port` phía server và GENERATOR_BASE_URL phía client |
| Đổi dataset | INPUT_JSONL ở bước 3 hoặc trước run mới |
| Chọn tổng số samples | LIMIT: 1, 5, 20, ...; all/rỗng = toàn bộ |
| Số samples mỗi batch | BATCH_SIZE, hiện là 20 |
| Số request đồng thời | GENERATOR_CONCURRENCY và JUDGE_CONCURRENCY, hiện đều là 20 |
| Thinking theo vai trò | GENERATOR_THINKING_MODE và JUDGE_THINKING_MODE; hỗ trợ low/high/xhigh/max, 1–100, on/off, default/none, alias deepseek_vllm_*, qwen3_on/off |
| Đổi nơi lưu kết quả | OUTPUT_DIR; mỗi RUN_ID là một thư mục riêng |
| Tiếp tục run cũ | RUN_ID cũ + RESUME=1, giữ cấu hình snapshot |

Bạn có thể sửa các giá trị mặc định ngay đầu `scripts/run_application.sh` hoặc export ở terminal. Biến đã export ở terminal có ưu tiên cao hơn giá trị sau `:-` trong script.

`MAX_OUTPUT_TOKENS=32768`, `API_TIMEOUT_SECONDS=1800`, `API_MAX_RETRIES=3`, `JUDGE_JSON_RETRIES=2`, `RETRY_BACKOFF_SECONDS=2` là các giá trị dùng xuyên suốt hướng dẫn này. API keys không được lưu trong config.json. Mọi đường dẫn dữ liệu/kết quả mặc định được xây từ PROJECT_ROOT; môi trường Python được chọn bằng môi trường đang activate.
