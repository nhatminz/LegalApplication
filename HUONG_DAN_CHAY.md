# Hướng dẫn chạy Legal Application Benchmark trên B200

Ví dụ xuyên suốt dùng checkpoint local:

```text
/workspace/storage-shared/models/Qwen3-4B
```

Repository runtime:

```text
/workspace/storage-shared/nlp/minhpn19/LegalApplication
```

Bạn dùng môi trường `.venv` đã cài đầy đủ thư viện, đặt ở bất kỳ đâu. Các lệnh gọi `python` và `vllm` từ môi trường đã activate. Chạy các bước có inference trên B200.

Luồng chạy: host Qwen3-4B trong terminal 1 → cấu hình client ở terminal 2 → làm sạch/validate input → chạy thử → chạy full → đọc kết quả. Judge là endpoint đã được cung cấp, không cần host lại Judge trong hướng dẫn này.

## 1. Activate môi trường hiện có

Mở hai terminal trên B200. Trong **mỗi terminal**, activate môi trường của bạn. Nếu đã activate thì bỏ qua bước này.

Hai dòng dưới có **một giá trị cần thay**: đường dẫn môi trường thực tế của bạn; đây không phải đường dẫn checkpoint hay repository.

```bash
export LEGAL_ENV_DIR="/duong/dan/thuc/te/den/moi-truong-venv-cua-ban"
source "$LEGAL_ENV_DIR/bin/activate"
```

Các bước sau giả định terminal đã dùng đúng môi trường này.

## 2. Terminal 1: host checkpoint Qwen3-4B

Ví dụ dùng GPU số 0, port 8001, tên model trên API là `Qwen/Qwen3-4B`.

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/minhpn19/LegalApplication
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
export PROJECT_ROOT=/workspace/storage-shared/nlp/minhpn19/LegalApplication
cd "$PROJECT_ROOT"

# Input thực tế. Có thể thay dòng này bằng file JSONL lớn hơn của bạn.
export INPUT_JSONL="$PROJECT_ROOT/input/judgment_with_issues_sample.jsonl"

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

- Quét toàn bộ input, kể cả khi một lần chạy có LIMIT nhỏ.
- Ghi dòng không parse được JSON/UTF-8 vào `removed_input_lines.jsonl`, rồi **xóa dòng đó khỏi chính INPUT_JSONL** và tiếp tục với dòng sau.
- Giữ nguyên nội dung/thứ tự các dòng còn lại; LIMIT tính trên samples hợp lệ sau cleanup.
- Nếu JSON parse được nhưng thiếu trường bắt buộc, sai schema, trùng doc_name hoặc thiếu nội dung luật, giữ nguyên record và báo lỗi để bạn sửa.
- Không gọi generator hoặc Judge.

Tệp mẫu trong workspace đã được làm sạch từ 7 dòng xuống 6 records hợp lệ. Nếu bản sao trên B200 vẫn còn dòng 7 bị cắt, bước này sẽ tự log và xóa dòng lỗi đó. Với file có 6 records thì chỉ chạy được tối đa 6 samples; cần input có ít nhất 20 records hợp lệ để có batch đủ 20 samples.

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
# export INPUT_JSONL="$PROJECT_ROOT/input/TEN_FILE_DU_LIEU_THUC_TE.jsonl"

export FULL_RUN_ID="qwen3-4b-full-$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d-%H%M%S)-${RANDOM}"

MODE=validate-only LIMIT=all RUN_ID="${FULL_RUN_ID}-validate" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh" \
  && MODE=full LIMIT=all RUN_ID="$FULL_RUN_ID" RESUME=0 \
  bash "$PROJECT_ROOT/scripts/run_application.sh"
```

Lệnh full chỉ chạy khi validation thành công. Cấu hình đã export ở bước 3 sẽ xử lý mỗi batch tối đa 20 samples: tối đa 20 generator requests đồng thời, sau đó tối đa 20 Judge requests đồng thời. Batch cuối ít hơn 20 vẫn chạy hết. Không có giới hạn tổng mặc định 1000 samples.

Nếu muốn **tổng cộng 20 samples**, dùng LIMIT=20 trong một run mới. Nếu muốn **toàn bộ dataset nhưng mỗi lượt 20**, dùng LIMIT=all như trên.

Giới hạn concurrency là số request client có thể gửi đồng thời; số sequences server thực sự chạy tại một thời điểm còn phụ thuộc scheduler và KV cache.

## 8. Theo dõi tiến trình và đọc artifacts

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
export RESULT_DIR=/workspace/storage-shared/nlp/minhpn19/LegalApplication/outputs/TEN_RUN_THUC_TE
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

Activate môi trường hiện có, đặt đường dẫn chính xác tới config.json của run cần resume, rồi copy khối dưới. Python chỉ chuyển các giá trị cấu hình đã lưu thành shell exports, không gọi API hoặc kiểm tra phiên bản thư viện.

```bash
export PROJECT_ROOT=/workspace/storage-shared/nlp/minhpn19/LegalApplication
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
| JSON parse được nhưng thiếu facts_clean/issues/luật hoặc doc_name trùng | Giữ record, báo lỗi schema và dừng preflight |
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
