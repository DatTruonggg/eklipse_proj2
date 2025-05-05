# LLM deployment - BLOOMZ-1b1 on-premise


## Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Solution Overview](#solution-overview)  
3. [Quick Run](#quick-run)  
4. [Optimization Techniques](#optimization-techniques)  
5. [API Endpoint](#api-endpoint)  
6. [Monitoring result](#monitoring-result)  
7. [Sample Prompts for Testing](#sample-prompts-for-testing)  

---


---

## Problem Statement

We aim to reduce costs of using cloud-based LLMs by deploying an **on-premise open-source LLM** (BLOOMZ-1b1).  
To make this efficient, we track:

- **RAM** usage  
- **Speed** (tokens per second)  
- **Performance** (accuracy or consistency)  
- **Feature Support** (log-probabilities)

---

## Solution Overview

We deploy `bloomz-1b1` using FastAPI on **CPU**, track metrics using **Prometheus**, and visualize them in **Grafana**.  
The model is served through a RESTful API with `/chat/completions` POST endpoint.

---

![alt text](/assets/prometheus-targer.png)

![alt text](image.png)

---

## Quick Run

### Setup 
1. Clone my github repo: [dattruong/eklipse_pro2](https://github.com/DatTruonggg/eklipse_proj2)


```cmd
git clone https://github.com/DatTruonggg/eklipse_proj2.git
```

2. Create new env with [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
```cmd 
make create-env
make activate-env 
```

```cmd
conda activate eklipse_pro2
```

- Please update your `HUGGINGFACE KEY` in `app/configs/config.yaml` :
``` yaml
logging_file: "./logs/logging_file.log"

model_name: "bigscience/bloomz-1b1"
embedding_model: "BAAI/llm-embedder"

llm_service: "http://localhost:8001"
max_token: 512
device: "cpu" #cuda or cpu
do_sample: true
skip_special_tokens: true
temperature: 0.1
top_p: 0.95
frequency_penalty: 0.1
stream: False

huggingface_key: "YOUR HUGGING FACE KEY"
```

### Docker
```cmd 
make docker-all
```

![docker-desktop-after-run%20](/assets/docker-desktop-after-run%20.png)

![docker-terminal](/assets/docker-terminal.png)

### UI 

- Open `http://localhost:8001/docs`, `http://localhost:9090/`, `http://localhost:3000/` to see the UI of **FastAPI**, **Prometheus**, **Grafana**

- FastUI
![alt text](/assets/fastapi.png)

- Prometheus UI with `/query` endpoint:

![alt text](/assets/prometheus-ui.png)

- Grafana login with the `user` and `password` = `admin`:

```
- GF_SECURITY_ADMIN_USER=admin
- GF_SECURITY_ADMIN_PASSWORD=admin
```
![Grafana-login](/assets/grafana-login.png)

- After login, we will see the Grafana UI as below:

![grafana-ui](/assets/grafana-ui.png)

### Setup new Grafana Dashboard
- **Step 1: Add a Data Source**
    1. From the left sidebar, go to **Connections → Data sources**
    2. Click **Add data source**
    3. Select **Prometheus** from the list of available data sources

- **Step 2: Configure Prometheus Settings**

    In the **Settings** tab:

    * **Name**: `prometheus` (or leave as default)

    * **Prometheus server URL**:
    If you're using Docker Compose:

    ```http
    http://prometheus:9090
    ```

    If running Prometheus locally:

    ```http
    http://localhost:9090
    ```

    * **Authentication**: Keep it as **No authentication** (unless you've enabled auth)

    ![alt text](/assets/setup-grafana-dashboard.png)

- **Step 3: Save & Test Connection**
    1. Click **Save & Test**
    2. You should see a success message: `Data source is working`

- **Step 4: Create Your First Dashboard**
    1. Go to **Dashboards** → **New Dashboard**
    2. Click **Add new panel**
    3. Enter a sample PromQL query, for example:
        ```cmd
        llm_tokens_per_second
        ```
    4. Choose your preferred visualization (Line, Bar, Gauge, etc.)
    5. Click Apply to save the panel


---

## Optimization Techniques

This section explains the techniques applied to optimize the inference performance of the `bloomz-1b1` model running on CPU.


```python
# LLM loading
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    trust_remote_code=True
)
model.eval()
```

```python
# Embedding model loading
tokenizer = AutoTokenizer.from_pretrained(embed_model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    embed_model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
model.eval()
```

### **Why Not Use ONNX?**

Although ONNX Runtime is known for speedup in inference, it is not suitable for this setup due to the following:

* **Compatibility Issues:** The `bloomz-1b1` model is not officially exported or supported in ONNX format on HuggingFace. Attempting to convert it leads to either export failures or loss of fidelity.
* **Tooling Overhead:** Using `optimum` library introduces extra dependencies (like `onnxruntime`, `optimum[onnxruntime]`, etc.) which did not justify the benefits in this specific case.
* **Deployment Simplicity:** Avoiding ONNX kept the setup clean and easily containerized within a FastAPI service.

### **Why Not Use Quantization?**

Quantization (e.g., using `torch.quantization.quantize_dynamic`) was tested but disabled due to the following:

* **Model Size:** `bloomz-1b1` is relatively small (1.1B parameters), and the gain from quantization on CPU was minimal.
* **Stability Problems:** Quantization led to FastAPI not launching properly in some cases, likely due to model incompatibility with certain dynamic quantization operations on specific layers.
* **Output Degradation:** Quantization introduced occasional drop in generation coherence or accuracy.

### Summary
- I tried using ONNX and quantization, but it was not successful.

![alt text](/assets/onnx_quantize.png)

| Technique         | Status    | Notes                                                             |
| ----------------- | --------- | ----------------------------------------------------------------- |
| Float32 Inference | Used    | Best balance of performance, accuracy and stability               |
| ONNX Runtime      | Skipped | Not supported natively for bloomz-1b1 + extra dependency overhead |
| Quantization      | Skipped | Caused FastAPI to fail and reduced output consistency             |

---

## API Endpoint
- The LLM service exposes a RESTful endpoint to interact with the deployed BLOOMZ-1b1 model.
- For example: 
![api-endpoint](/assets/api-endpoint.png)
![response](/assets/response.png)
### **POST /chat/completions**

This endpoint accepts a JSON payload that contains a list of chat messages and generation parameters.

#### Example Request

```http
POST http://localhost:8001/chat/completions
Content-Type: application/json
```

```json
{
  "model": "bigscience/bloomz-1b1",
  "messages": [
    {
      "role": "user",
      "content": "What are the benefits of renewable energy?"
    }
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "frequency_penalty": 0,
  "stream": false
}
```

#### Example Response

```json
{
  "id": "completion-id",
  "object": "chat.completion",
  "created": 1715000000.123,
  "model": "bigscience/bloomz-1b1",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Renewable energy helps reduce carbon emissions, improves public health, and promotes energy independence..."
      }
    }
  ]
}
```

#### Notes

* The API is built using FastAPI.
* It supports both streaming and non-streaming responses.
* Metrics such as latency, token/sec, and RAM usage are automatically captured and exported to Prometheus.

---

## Monitoring result
- Example result:

![grafana-result](/assets/grafana-result.png)

## Monitoring Result

Below are the monitoring results of the deployed Bloomz-1b1 model, visualized using **Grafana** and tracked via **Prometheus** metrics. Each graph provides insights into a specific performance dimension:

### Speed
- **Metric:** `llm_tokens_per_second`
- This graph shows how many tokens the model generates per second.
- Higher values indicate faster inference.
- We observed an increase in generation speed after applying model loading optimizations (e.g., setting `torch_dtype=torch.float32` explicitly).

### RAM Usage
- **Metric:** `llm_ram_usage_mb`
- Tracks the memory consumption (in MB) of the FastAPI app hosting the LLM.
- Spikes appear during model loading or when handling concurrent requests.
- RAM usage remained relatively stable (~4800MB) after stabilization.

### Avg Latency per Request
- **Metric:** `llm_request_latency_seconds`
- Measures the average response time per LLM API request.
- Initial latency was high due to cold start and model initialization.
- Latency drops and stabilizes once the model is loaded and running.

### Performance
- **Metric:** `llm_eval_accuracy`
- Static metric reflecting the model's evaluation accuracy on benchmark tasks.
- For Bloomz-1b1, we set this manually at `0.71` based on public benchmarks.
- This is useful for reference when comparing across models.

### Summary
- Monitoring dashboards confirm that the model serves requests reliably with acceptable latency and resource usage.
- While ONNX or quantization was considered, native PyTorch with `float32` yielded the best balance of stability and performance for this CPU-based deployment.

---

## Sample Prompts for Testing

Here are 10 sample prompts used to evaluate and stress-test the Bloomz-1b1 model during inference monitoring:

1. **Knowledge question:**  
   *"What is the capital city of Australia?"*

2. **Creative writing:**  
   *"Write a short poem about the moon and the sea."*

3. **Reasoning:**  
   *"If it is raining in Paris and you are carrying an umbrella, will you get wet?"*

4. **Math problem:**  
   *"What is the square root of 144, and how is it calculated?"*

5. **Translation:**  
   *"Translate the sentence 'I love programming' into French."*

6. **Instruction following:**  
   *"List three tips to improve your concentration while studying."*

7. **Comparison task:**  
   *"Compare the benefits of green tea and black coffee."*

8. **Summarization:**  
   *"Summarize the following paragraph in one sentence: 'Artificial intelligence is transforming industries across the world...'"*

9. **Open-ended chat:**  
   *"Hi, how are you today?"*

10. **Programming help:**  
   *"Write a Python function that checks whether a number is a palindrome."*
