const modelData = [
  { model_name: "bert_llrd_weighted_ce", accuracy: 0.6825, macro_f1: 0.5646, weighted_f1: 0.6891, precision: 0.5636, recall: 0.5690, mae: 0.3505, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7528 },
  { model_name: "bert_full_undersample_ce", accuracy: 0.6563, macro_f1: 0.5490, weighted_f1: 0.6671, precision: 0.5296, recall: 0.5790, mae: 0.3932, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7489 },
  { model_name: "bert_full_weighted_ce", accuracy: 0.6772, macro_f1: 0.5471, weighted_f1: 0.6851, precision: 0.5632, recall: 0.5449, mae: 0.3508, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7471 },
  { model_name: "distilbert_full_weighted_ce", accuracy: 0.6707, macro_f1: 0.5421, weighted_f1: 0.6772, precision: 0.5352, recall: 0.5507, mae: 0.3688, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3772 },
  { model_name: "distilbert_llrd_weighted_ce", accuracy: 0.6630, macro_f1: 0.5409, weighted_f1: 0.6733, precision: 0.5453, recall: 0.5470, mae: 0.3773, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3862 },
  { model_name: "bert_llrd_undersample_ce", accuracy: 0.6580, macro_f1: 0.5247, weighted_f1: 0.6615, precision: 0.5190, recall: 0.5424, mae: 0.3882, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7524 },
  { model_name: "distilbert_llrd_undersample_ce", accuracy: 0.6345, macro_f1: 0.5205, weighted_f1: 0.6472, precision: 0.4996, recall: 0.5565, mae: 0.4315, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3857 },
  { model_name: "distilbert_full_undersample_ce", accuracy: 0.6306, macro_f1: 0.5181, weighted_f1: 0.6432, precision: 0.4971, recall: 0.5537, mae: 0.4348, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3821 },
  { model_name: "bilstm_attention_weighted_ce", accuracy: 0.6356, macro_f1: 0.4921, weighted_f1: 0.6421, precision: 0.5085, recall: 0.4860, mae: 0.4150, num_parameters: 3998213, model_size_mb: 15.25, inference_time_per_sample_ms: 0.5938 },
  { model_name: "bilstm_weighted_ce", accuracy: 0.6524, macro_f1: 0.4854, weighted_f1: 0.6511, precision: 0.4948, recall: 0.4783, mae: 0.4118, num_parameters: 3997701, model_size_mb: 15.25, inference_time_per_sample_ms: 0.5860 },
  { model_name: "bilstm_attention_undersample_ce", accuracy: 0.5470, macro_f1: 0.4354, weighted_f1: 0.5692, precision: 0.4351, recall: 0.4515, mae: 0.5523, num_parameters: 3166725, model_size_mb: 12.08, inference_time_per_sample_ms: 0.5934 },
  { model_name: "bilstm_undersample_ce", accuracy: 0.5432, macro_f1: 0.3825, weighted_f1: 0.5544, precision: 0.3779, recall: 0.4253, mae: 0.6789, num_parameters: 3166213, model_size_mb: 12.08, inference_time_per_sample_ms: 0.5879 },
  { model_name: "distilbert_freeze_weighted_ce", accuracy: 0.5490, macro_f1: 0.2826, weighted_f1: 0.5302, precision: 0.2901, recall: 0.2961, mae: 0.6071, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3827 },
  { model_name: "bert_freeze_weighted_ce", accuracy: 0.4901, macro_f1: 0.2525, weighted_f1: 0.4682, precision: 0.2601, recall: 0.2607, mae: 0.7735, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7354 },
  { model_name: "bert_freeze_undersample_ce", accuracy: 0.3711, macro_f1: 0.2267, weighted_f1: 0.4032, precision: 0.2363, recall: 0.2363, mae: 1.2006, num_parameters: 109486085, model_size_mb: 417.66, inference_time_per_sample_ms: 6.7392 },
  { model_name: "distilbert_freeze_undersample_ce", accuracy: 0.4309, macro_f1: 0.1944, weighted_f1: 0.4033, precision: 0.2634, recall: 0.2353, mae: 0.9458, num_parameters: 66366725, model_size_mb: 253.17, inference_time_per_sample_ms: 3.3818 }
];

const ensembleAlphaData = [
  { alpha: 0.3, accuracy: 0.6851, macro_f1: 0.5589, weighted_f1: 0.6929, precision: 0.5717, recall: 0.5582, mae: 0.3440 },
  { alpha: 0.4, accuracy: 0.6863, macro_f1: 0.5599, weighted_f1: 0.6943, precision: 0.5681, recall: 0.5612, mae: 0.3434 },
  { alpha: 0.5, accuracy: 0.6834, macro_f1: 0.5647, weighted_f1: 0.6917, precision: 0.5660, recall: 0.5688, mae: 0.3487 },
  { alpha: 0.6, accuracy: 0.6722, macro_f1: 0.5579, weighted_f1: 0.6817, precision: 0.5492, recall: 0.5700, mae: 0.3641 },
  { alpha: 0.7, accuracy: 0.6645, macro_f1: 0.5498, weighted_f1: 0.6747, precision: 0.5355, recall: 0.5694, mae: 0.3785 }
];

const robustnessData = [
  { model_name: "BiLSTM", clean_accuracy: 0.6524, noisy_accuracy: 0.6194, drop_accuracy: 0.0330, clean_macro_f1: 0.4854, noisy_macro_f1: 0.4280, drop_macro_f1: 0.0574, clean_mae: 0.4118, noisy_mae: 0.4636, drop_mae: 0.0518 },
  { model_name: "BiLSTM+Attention", clean_accuracy: 0.6356, noisy_accuracy: 0.6024, drop_accuracy: 0.0333, clean_macro_f1: 0.4921, noisy_macro_f1: 0.4395, drop_macro_f1: 0.0525, clean_mae: 0.4150, noisy_mae: 0.4574, drop_mae: 0.0424 },
  { model_name: "DistilBERT", clean_accuracy: 0.6707, noisy_accuracy: 0.6663, drop_accuracy: 0.0044, clean_macro_f1: 0.5421, noisy_macro_f1: 0.5228, drop_macro_f1: 0.0193, clean_mae: 0.3688, noisy_mae: 0.3944, drop_mae: 0.0256 },
  { model_name: "BERT-base", clean_accuracy: 0.6825, noisy_accuracy: 0.6736, drop_accuracy: 0.0088, clean_macro_f1: 0.5646, noisy_macro_f1: 0.5322, drop_macro_f1: 0.0324, clean_mae: 0.3505, noisy_mae: 0.3723, drop_mae: 0.0218 }
];

const errorSummaryData = [
  { model_family: "bilstm", checkpoint: "bilstm_weighted_ce", accuracy: 0.6524, mae: 0.4118, error_rate: 0.3476 },
  { model_family: "bilstm_attn", checkpoint: "bilstm_attention_weighted_ce", accuracy: 0.6356, mae: 0.4150, error_rate: 0.3644 },
  { model_family: "distilbert", checkpoint: "distilbert_full_weighted_ce", accuracy: 0.6707, mae: 0.3688, error_rate: 0.3293 },
  { model_family: "bert", checkpoint: "bert_llrd_weighted_ce", accuracy: 0.6825, mae: 0.3505, error_rate: 0.3175 }
];

const errorCategoryFocus = [
  { category: "subtle_rating_difference", count: 45 },
  { category: "mixed_sentiment", count: 4 },
  { category: "ambiguous_review", count: 1 }
];

const splitStats = [
  { rating: 1, train: 575, val: 123, test: 123 },
  { rating: 2, train: 1084, val: 233, test: 232 },
  { rating: 3, train: 1976, val: 423, test: 424 },
  { rating: 4, train: 3435, val: 736, test: 736 },
  { rating: 5, train: 8769, val: 1879, test: 1880 }
];

const sampleTextData = [
  {
    id: "s1",
    rating: 1,
    title: "Wrinkled!",
    department: "Dresses",
    class_name: "Dresses",
    review_text: "The dress arrived wrinkled so i washed it and pressed it. it looked beautiful until i sat down. the entire dress was wrinkled and it looked terrible all day. can't wear this again."
  },
  {
    id: "s2",
    rating: 2,
    title: "Prettier in pic!",
    department: "Tops",
    class_name: "Blouses",
    review_text: "I just returned this top. i agree with the previous reviewer. the neckline is way too deep, i ordered small and i looked ridiculous in it. also the ruffles didn't look so great on me either."
  },
  {
    id: "s3",
    rating: 3,
    title: "As others have said...",
    department: "Dresses",
    class_name: "Dresses",
    review_text: "This is a very fetching little dress but the details of the fit, as described by other reviewers, is slightly off. i am 34d 5'5 and 140 lbs. the top fits perfectly but has an odd gap at the button placket."
  },
  {
    id: "s4",
    rating: 4,
    title: "Pretty but runs long",
    department: "Tops",
    class_name: "Blouses",
    review_text: "I bought a size xl in the white with brighter small florals. the details on the top are just as nice in person as shown in the photos. however, it runs long and i need to belt it or tuck it."
  },
  {
    id: "s5",
    rating: 5,
    title: "Amazing dress!",
    department: "Dresses",
    class_name: "Dresses",
    review_text: "Wow! i saw this dress in the store and had to try it on. it is simply stunning! the bright colors and design are a true tracy reese dress and i looked like a fairy queen in it! lol. i can't wait to wear it!"
  }
];

const caseStudies = [
  {
    tag: "Case đúng",
    source: "Verified inference with bert_llrd_weighted_ce",
    title: "Amazing dress!",
    review_text: "Wow! i saw this dress in the store and had to try it on. it is simply stunning! the bright colors and design are a true tracy reese dress and i looked like a fairy queen in it! lol. i can't wait to wear it!",
    true_rating: 5,
    pred_rating: 5,
    confidence: 0.9344,
    note: "Sentiment rất rõ ràng, model dự đoán đúng và confidence cao."
  },
  {
    tag: "Case mixed",
    source: "outputs/reports/error_analysis/bert_llrd_weighted_ce_misclassified.csv",
    title: "This is a zippered hoodie",
    review_text: "Just wanted to review so people know this hoodie has a zipper. it's very soft and comfy but i was looking for a pullover hoodie and am disappointed that once again retailer's picture doesn't match the actual product.",
    true_rating: 1,
    pred_rating: 3,
    confidence: null,
    note: "Văn bản có tín hiệu trái chiều (khen chất liệu nhưng thất vọng kỳ vọng), dễ gây lệch rating."
  },
  {
    tag: "Case nghi ngờ nhãn",
    source: "outputs/reports/error_analysis/bert_llrd_weighted_ce_misclassified.csv",
    title: "Not as hoped",
    review_text: "I wished these fit a little better. unfortunately, they were a little tight in the thighs and if i didn't pull the waist of the pants up, i had sagging crotch and the pockets hung out.",
    true_rating: 1,
    pred_rating: 3,
    confidence: null,
    note: "Được gán error_category = ambiguous_review; văn bản thiếu cường độ tiêu cực mạnh so với nhãn 1."
  }
];

const bertIgCases = [
  {
    id: 0,
    title: "Amazing dress!",
    status: "correct",
    text_line: "Amazing dress! Wow! i saw this dress in the store and had to try it on. it is simply stunning! the bright colors and design are a true tracy reese dress and i looked like a fairy queen in it! lol. i can't wait to wear it!",
    true_rating: 5,
    pred_rating: 5,
    confidence: 0.9344,
    source: "bert_xai.html",
    top_tokens: [
      { token: "[SEP]", weight: 1.0 },
      { token: "[SEP]", weight: 0.8050 },
      { token: "stunning", weight: 0.2816 },
      { token: "wait", weight: 0.1463 },
      { token: "amazing", weight: 0.1348 }
    ],
    insight: "IG tập trung vào từ cảm xúc mạnh (stunning, amazing) nên dự đoán lớp 5 ổn định và độ tin cậy cao."
  },
  {
    id: 2,
    title: "Pretty but runs long",
    status: "correct",
    text_line: "Pretty but runs long. the details on the top are just as nice in person as shown in the photos. however, it runs long and i need to belt it or tuck it.",
    true_rating: 4,
    pred_rating: 4,
    confidence: 0.7400,
    source: "bert_xai.html",
    top_tokens: [
      { token: "[SEP]", weight: 1.0 },
      { token: "love", weight: 0.2218 },
      { token: "well", weight: 0.1934 },
      { token: "love", weight: 0.1925 },
      { token: "[SEP]", weight: 0.1414 }
    ],
    insight: "Nội dung vừa khen vừa góp ý, nhưng các token tích cực (love, well) chiếm trọng số cao nên mô hình giữ đúng lớp 4."
  },
  {
    id: 3,
    title: "Pretty blouse",
    status: "correct",
    text_line: "Pretty blouse. very cute but a little short. currently pregnant so i'm hoping it will hang better in a few months.",
    true_rating: 4,
    pred_rating: 4,
    confidence: 0.8115,
    source: "bert_xai.html",
    top_tokens: [
      { token: "[SEP]", weight: 1.0 },
      { token: "[SEP]", weight: 0.3758 },
      { token: "better", weight: 0.3752 },
      { token: "blouse", weight: 0.3284 },
      { token: "pretty", weight: 0.3255 }
    ],
    insight: "IG nhấn mạnh các token mô tả mức hài lòng vừa phải (pretty, better), phù hợp với dự đoán rating 4."
  },
  {
    id: 1,
    title: "Well fitting bra, comfortable",
    status: "incorrect",
    text_line: "Well fitting bra, comfortable (not itchy), lace doesn't show through tops.",
    true_rating: 5,
    pred_rating: 4,
    confidence: 0.7864,
    source: "bert_xai.html",
    top_tokens: [
      { token: "comfortable", weight: 1.0 },
      { token: "well", weight: 0.7420 },
      { token: "[SEP]", weight: 0.5735 },
      { token: "through", weight: 0.3560 },
      { token: "not", weight: 0.2485 }
    ],
    insight: "Model nhận mạnh tín hiệu tích cực (comfortable, well) nhưng vẫn có token phủ định (not), nên lệch xuống lớp 4 thay vì 5."
  },
  {
    id: 5,
    title: "Prettier in pic!",
    status: "incorrect",
    text_line: "Prettier in pic! i just returned this top. the neckline is way too deep, i ordered small and i looked ridiculous in it. also the ruffles didn't look so great on me either.",
    true_rating: 2,
    pred_rating: 3,
    confidence: 0.7198,
    source: "bert_xai.html",
    top_tokens: [
      { token: "[SEP]", weight: 1.0 },
      { token: "##tti", weight: 0.8757 },
      { token: "[SEP]", weight: 0.7582 },
      { token: "pre", weight: 0.7301 },
      { token: "too", weight: 0.5582 }
    ],
    insight: "Các token nổi bật bị tách subword mạnh (pre + ##tti) và thiên về tín hiệu trung tính/miêu tả, khiến dự đoán dồn về lớp giữa (3)."
  },
  {
    id: 7,
    title: "Great summer dress",
    status: "incorrect",
    text_line: "Great summer dress. super comfortable summer dress. just bought it yesterday and i don't want to take it off. soft lightweight fabric.",
    true_rating: 5,
    pred_rating: 4,
    confidence: 0.5359,
    source: "bert_xai.html",
    top_tokens: [
      { token: "[SEP]", weight: 1.0 },
      { token: "[SEP]", weight: 0.7793 },
      { token: "comfortable", weight: 0.7121 },
      { token: "'", weight: 0.5607 },
      { token: "dress", weight: 0.3984 }
    ],
    insight: "Dù có tín hiệu tích cực, confidence thấp và điểm chú ý phân tán nên mô hình chỉ dự đoán lớp lân cận 4 thay vì 5."
  }
];

const demoCases = [
  {
    id: "demo-correct",
    label: "Case dung - Amazing dress! (true=5, pred=5)",
    title: "Amazing dress!",
    review_text: "Wow! i saw this dress in the store and had to try it on. it is simply stunning! the bright colors and design are a true tracy reese dress and i looked like a fairy queen in it! lol. i can't wait to wear it!",
    true_rating: 5,
    pred_rating: 5,
    confidence: 0.9344,
    source: "bert_llrd_weighted_ce inference"
  },
  {
    id: "demo-subtle",
    label: "Case subtle error - Wrinkled! (true=1, pred=2)",
    title: "Wrinkled!",
    review_text: "The dress arrived wrinkled so i washed it and pressed it. it looked beautiful until i sat down. the entire dress was wrinkled and it looked terrible all day. can't wear this again.",
    true_rating: 1,
    pred_rating: 2,
    confidence: null,
    source: "error_analysis misclassified"
  },
  {
    id: "demo-mixed",
    label: "Case mixed sentiment (true=1, pred=3)",
    title: "This is a zippered hoodie",
    review_text: "Just wanted to review so people know this hoodie has a zipper. it's very soft and comfy but i was looking for a pullover hoodie and am disappointed that once again retailer's picture doesn't match the actual product.",
    true_rating: 1,
    pred_rating: 3,
    confidence: null,
    source: "error_analysis misclassified"
  },
  {
    id: "demo-ambiguous",
    label: "Case ambiguous_review (true=1, pred=3)",
    title: "Not as hoped",
    review_text: "I wished these fit a little better. unfortunately, they were a little tight in the thighs and if i didn't pull the waist of the pants up, i had sagging crotch and the pockets hung out.",
    true_rating: 1,
    pred_rating: 3,
    confidence: null,
    source: "error_analysis misclassified"
  }
];

const confusionImages = [
  "bert_freeze_undersample_ce",
  "bert_freeze_weighted_ce",
  "bert_full_undersample_ce",
  "bert_full_weighted_ce",
  "bert_llrd_undersample_ce",
  "bert_llrd_weighted_ce",
  "bilstm_attention_undersample_ce",
  "bilstm_attention_weighted_ce",
  "bilstm_undersample_ce",
  "bilstm_weighted_ce",
  "distilbert_freeze_undersample_ce",
  "distilbert_freeze_weighted_ce",
  "distilbert_full_undersample_ce",
  "distilbert_full_weighted_ce",
  "distilbert_llrd_undersample_ce",
  "distilbert_llrd_weighted_ce"
];

const curveShowcase = [
  { model: "bilstm_weighted_ce", family: "BiLSTM" },
  { model: "bilstm_attention_weighted_ce", family: "BiLSTM+Attention" },
  { model: "distilbert_full_weighted_ce", family: "DistilBERT" },
  { model: "bert_llrd_weighted_ce", family: "BERT-base" }
];

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function normToken(token) {
  return String(token || "")
    .toLowerCase()
    .replace(/^#+/, "")
    .replace(/[^a-z0-9]/g, "");
}

function igColor(weight) {
  const w = Math.max(0, Math.min(1, Number(weight) || 0));
  const r = Math.round(255 * w);
  const g = Math.round(255 - 128 * w);
  const b = 100;
  return `rgb(${r},${g},${b})`;
}

function renderIgLine(text, topTokens) {
  const tokenMap = new Map();
  topTokens.forEach((t) => {
    const key = normToken(t.token);
    if (!key || key === "sep" || key.length < 2) return;
    tokenMap.set(key, Math.max(t.weight, tokenMap.get(key) || 0));
  });

  const parts = String(text || "").split(/(\s+)/);
  return parts
    .map((part) => {
      if (/^\s+$/.test(part)) return part;

      const key = normToken(part);
      if (!key) return escapeHtml(part);

      let weight = tokenMap.get(key) || 0;
      if (!weight && key.length > 4) {
        tokenMap.forEach((v, k) => {
          if (k.length > 3 && (key.includes(k) || k.includes(key))) {
            weight = Math.max(weight, v);
          }
        });
      }

      if (!weight) {
        return `<span class="ig-word">${escapeHtml(part)}</span>`;
      }

      return `<span class="ig-word ig-word-hit" style="background-color:${igColor(weight)}" title="weight=${fmt(weight, 4)}">${escapeHtml(part)}</span>`;
    })
    .join("");
}

let bertIgRawMapPromise = null;

function loadBertIgRawMap() {
  if (bertIgRawMapPromise) return bertIgRawMapPromise;

  bertIgRawMapPromise = fetch("../outputs/reports/xai_results/bert_xai.html")
    .then((resp) => {
      if (!resp.ok) throw new Error(`Cannot load bert_xai.html (${resp.status})`);
      return resp.text();
    })
    .then((html) => {
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, "text/html");
      const blocks = [...doc.querySelectorAll("div[style*='font-family:monospace']")];
      const map = new Map();

      blocks.forEach((block) => {
        const header = block.querySelector("p");
        if (!header) return;

        const idMatch = header.textContent.match(/ID:\s*(\d+)/i);
        if (!idMatch) return;
        const id = Number(idMatch[1]);

        const lineHtml = block.innerHTML.replace(header.outerHTML, "").trim();
        const topTokensNode = block.nextElementSibling && block.nextElementSibling.tagName === "P"
          ? block.nextElementSibling
          : null;
        const topTokensHtml = topTokensNode ? topTokensNode.outerHTML : "";

        map.set(id, { lineHtml, topTokensHtml });
      });

      return map;
    })
    .catch(() => new Map());

  return bertIgRawMapPromise;
}

function fmtInt(value) {
  return Number(value).toLocaleString("en-US");
}

function truncateText(text, maxLen = 145) {
  const raw = String(text || "").trim();
  if (raw.length <= maxLen) return raw;
  return `${raw.slice(0, maxLen - 1)}...`;
}

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function tokenOverlapScore(a, b) {
  const setA = new Set(tokenize(a));
  const setB = new Set(tokenize(b));
  if (!setA.size || !setB.size) return 0;
  let inter = 0;
  setA.forEach((t) => {
    if (setB.has(t)) inter += 1;
  });
  return inter / Math.max(setA.size, setB.size);
}

function getFamilyName(modelName) {
  if (modelName.startsWith("bilstm_attention")) return "bilstm_attention";
  if (modelName.startsWith("bilstm")) return "bilstm";
  if (modelName.startsWith("distilbert")) return "distilbert";
  if (modelName.startsWith("bert")) return "bert";
  return "other";
}

function createTable(headers, rows) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");

  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  return table;
}

function buildRankMap(rows, key, goal, topN = 3) {
  const ranked = [...rows]
    .sort((a, b) => (goal === "max" ? b[key] - a[key] : a[key] - b[key]))
    .slice(0, topN);

  const rankMap = new Map();
  ranked.forEach((row, idx) => {
    rankMap.set(row, idx + 1);
  });
  return rankMap;
}

function createRankedMetricTable(columns, rows) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");

  const rankMaps = {};
  columns
    .filter((col) => col.goal)
    .forEach((col) => {
      rankMaps[col.key] = buildRankMap(rows, col.key, col.goal, 3);
    });

  columns.forEach((col) => {
    const th = document.createElement("th");

    if (!col.goal) {
      th.textContent = col.label;
      trHead.appendChild(th);
      return;
    }

    const headWrap = document.createElement("span");
    headWrap.className = "metric-head";

    const label = document.createElement("span");
    label.textContent = col.label;

    const goal = document.createElement("span");
    goal.className = `metric-goal ${col.goal === "max" ? "up" : "down"}`;
    goal.textContent = col.goal === "max" ? "↑" : "↓";
    goal.title = col.goal === "max" ? "Cang cao cang tot" : "Cang thap cang tot";
    goal.setAttribute("aria-label", goal.title);

    headWrap.appendChild(label);
    headWrap.appendChild(goal);
    th.appendChild(headWrap);
    if (col.dividerAfter) th.classList.add("metric-divider");
    trHead.appendChild(th);
  });

  thead.appendChild(trHead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");

    columns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = col.formatter ? col.formatter(row[col.key]) : String(row[col.key]);
      if (col.dividerAfter) td.classList.add("metric-divider");

      if (col.goal) {
        const rank = rankMaps[col.key].get(row);
        if (rank) {
          td.classList.add("metric-rank", `metric-rank-${rank}`);
          td.title = `Top ${rank} cho ${col.label}`;
        }
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  return table;
}

function renderHeroMeta() {
  const wrap = document.getElementById("heroMeta");
  if (!wrap) return;
  const items = [
    ["Dataset", "Womens Clothing E-Commerce Reviews"],
    ["Framework", "PyTorch + Transformers"],
    ["Theo dõi", "WandB + local outputs"],
    ["Input", "title + review_text"],
    ["Output", "rating 1..5"]
  ];

  wrap.innerHTML = "";
  items.forEach(([k, v]) => {
    const chip = document.createElement("span");
    chip.className = "meta-chip";
    chip.innerHTML = `<strong>${k}</strong><span>${v}</span>`;
    wrap.appendChild(chip);
  });
}

function renderHeroHighlights() {
  const wrap = document.getElementById("heroHighlights");
  if (!wrap) return;

  const bestMacro = [...modelData].sort((a, b) => b.macro_f1 - a.macro_f1)[0];
  const bestAcc = [...modelData].sort((a, b) => b.accuracy - a.accuracy)[0];
  const configCount = modelData.length;
  const avgLatency = modelData.reduce((sum, x) => sum + x.inference_time_per_sample_ms, 0) / modelData.length;

  const lines = [
    `Checkpoint tốt nhất tổng thể: ${bestMacro.model_name}`,
    `Best Macro-F1: ${fmt(bestMacro.macro_f1)} | Best Accuracy: ${fmt(bestAcc.accuracy)}`,
    `Tổng ${configCount} cấu hình được benchmark trên test split`,
    `Latency trung bình toàn bộ run: ${fmt(avgLatency, 3)} ms/sample`
  ];

  wrap.innerHTML = `<ul class="highlight-list">${lines.map((line) => `<li>${line}</li>`).join("")}</ul>`;
}

function renderKpis() {
  const wrap = document.getElementById("kpiCards");
  if (!wrap) return;

  const bestMacro = [...modelData].sort((a, b) => b.macro_f1 - a.macro_f1)[0];
  const bestAcc = [...modelData].sort((a, b) => b.accuracy - a.accuracy)[0];
  const bestMae = [...modelData].sort((a, b) => a.mae - b.mae)[0];
  const fastest = [...modelData].sort((a, b) => a.inference_time_per_sample_ms - b.inference_time_per_sample_ms)[0];
  const cards = [
    { label: "Best Macro-F1", value: fmt(bestMacro.macro_f1), note: bestMacro.model_name },
    { label: "Best Accuracy", value: fmt(bestAcc.accuracy), note: bestAcc.model_name },
    { label: "Best MAE", value: fmt(bestMae.mae), note: bestMae.model_name },
    { label: "Fastest ms/sample", value: fmt(fastest.inference_time_per_sample_ms, 3), note: fastest.model_name }
  ];

  wrap.innerHTML = "";
  cards.forEach((card) => {
    const box = document.createElement("article");
    box.className = "kpi-card";
    box.innerHTML = `<p>${card.label}</p><strong>${card.value}</strong><p>${card.note}</p>`;
    wrap.appendChild(box);
  });
}

function renderProblemStats() {
  const wrap = document.getElementById("problemStatCards");
  if (!wrap) return;

  const totalRows = splitStats.reduce((sum, row) => sum + row.train + row.val + row.test, 0);
  const rawRows = 23486;
  const cards = [
    { title: "Loại bài toán", value: "Text multiclass classification", note: "Phân loại rating 5 lớp" },
    { title: "Số lớp", value: "5", note: "Ratings 1 -> 5" },
    { title: "Core metrics", value: "Macro-F1 / Accuracy / MAE", note: "Đánh giá chất lượng và ordinal error" },
    { title: "Số cấu hình chạy", value: `${modelData.length}`, note: `Dữ liệu gốc ban đầu: ${fmtInt(rawRows)} mẫu (sau clean: ${fmtInt(totalRows)})` }
  ];

  wrap.innerHTML = "";
  cards.forEach((card) => {
    const el = document.createElement("article");
    el.className = "stat-card";
    el.innerHTML = `<p>${card.title}</p><strong>${card.value}</strong><p>${card.note}</p>`;
    wrap.appendChild(el);
  });
}

function renderSampleCards() {
  const wrap = document.getElementById("sampleCards");
  if (!wrap) return;

  wrap.innerHTML = "";
  sampleTextData.forEach((item) => {
    const card = document.createElement("article");
    card.className = "sample-card";
    card.innerHTML = `
      <div class="sample-head">
        <strong>Rating ${item.rating}</strong>
        <span>${item.department} / ${item.class_name}</span>
      </div>
      <p class="sample-title">${item.title || "(no title)"}</p>
      <p class="sample-text">${truncateText(item.review_text, 140)}</p>
      <button class="ghost-btn open-full-text" data-title="${(item.title || "Toàn văn").replace(/"/g, "&quot;")}" data-body="${item.review_text.replace(/"/g, "&quot;")}">Xem toàn văn</button>
    `;
    wrap.appendChild(card);
  });
}

function renderSplitTable() {
  const wrap = document.getElementById("splitTableWrap");
  if (!wrap) return;

  const trainTotal = splitStats.reduce((sum, row) => sum + row.train, 0);
  const valTotal = splitStats.reduce((sum, row) => sum + row.val, 0);
  const testTotal = splitStats.reduce((sum, row) => sum + row.test, 0);

  const rows = splitStats.map((r) => [
    String(r.rating),
    `${fmtInt(r.train)} (${fmt((r.train / trainTotal) * 100, 2)}%)`,
    `${fmtInt(r.val)} (${fmt((r.val / valTotal) * 100, 2)}%)`,
    `${fmtInt(r.test)} (${fmt((r.test / testTotal) * 100, 2)}%)`
  ]);
  rows.push(["Total", fmtInt(trainTotal), fmtInt(valTotal), fmtInt(testTotal)]);

  wrap.innerHTML = "";
  wrap.appendChild(createTable(["rating", "train", "val", "test"], rows));
}

function renderArchitectureFlow() {
  const wrap = document.getElementById("architectureFlow");
  if (!wrap) return;

  const steps = [
    "Raw CSV",
    "Làm sạch + full_text",
    "Chia stratified",
    "Dataloader RNN/Transformer",
    "Huấn luyện (orig + weighted CE / undersample + CE)",
    "Đánh giá + phân tích"
  ];

  wrap.innerHTML = "";
  steps.forEach((step, idx) => {
    const node = document.createElement("div");
    node.className = "flow-node";
    node.textContent = step;
    wrap.appendChild(node);

    if (idx < steps.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "flow-arrow";
      arrow.textContent = "->";
      wrap.appendChild(arrow);
    }
  });
}

function renderCurveGallery(containerId) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;

  wrap.innerHTML = "";
  curveShowcase.forEach((row) => {
    const fig = document.createElement("figure");
    fig.className = "figure-card";
    fig.innerHTML = `
      <img src="../outputs/figures/training_curves/${row.model}_training_curves.png" alt="${row.model} training curves" loading="lazy">
      <figcaption>${row.family} - ${row.model}_training_curves.png</figcaption>
    `;
    wrap.appendChild(fig);
  });
}

function renderLrConfig() {
  const el = document.getElementById("lrConfigCard");
  if (!el) return;
  el.innerHTML = `
    <ul>
      <li>Transformer LR mặc định: 2e-5, optimizer AdamW.</li>
      <li>Warmup ratio: 0.1, sau đó linear decay đến 0.</li>
      <li>LLRD mode được bật cho các run bert_llrd_* và distilbert_llrd_*.</li>
      <li>Learning rate được log qua wandb_run.log(learning_rate) theo epoch.</li>
    </ul>
  `;
}

function sortModels(data, key) {
  if (key === "model_name") {
    return [...data].sort((a, b) => String(a.model_name).localeCompare(String(b.model_name)));
  }

  const asc = key === "mae" || key === "num_parameters" || key === "inference_time_per_sample_ms" || key === "model_size_mb";
  return [...data].sort((a, b) => (asc ? a[key] - b[key] : b[key] - a[key]));
}

const modelTableColumns = [
  { key: "model_name", label: "model_name", goal: null, formatter: (v) => String(v) },
  { key: "accuracy", label: "accuracy", goal: "max", formatter: (v) => fmt(v) },
  { key: "macro_f1", label: "macro_f1", goal: "max", formatter: (v) => fmt(v) },
  { key: "weighted_f1", label: "weighted_f1", goal: "max", formatter: (v) => fmt(v) },
  { key: "precision", label: "precision", goal: "max", formatter: (v) => fmt(v) },
  { key: "recall", label: "recall", goal: "max", formatter: (v) => fmt(v) },
  { key: "mae", label: "mae", goal: "min", formatter: (v) => fmt(v) },
  { key: "num_parameters", label: "num_parameters", goal: "min", formatter: (v) => fmtInt(v) },
  { key: "model_size_mb", label: "model_size_mb", goal: "min", formatter: (v) => fmt(v, 2) },
  { key: "inference_time_per_sample_ms", label: "inference_ms", goal: "min", formatter: (v) => fmt(v, 4) }
];

function getTopRankMaps(rows, topN = 3) {
  const maps = {};

  modelTableColumns
    .filter((col) => col.goal)
    .forEach((col) => {
      const ranked = [...rows]
        .sort((a, b) => (col.goal === "max" ? b[col.key] - a[col.key] : a[col.key] - b[col.key]))
        .slice(0, topN);

      const rankByModel = new Map();
      ranked.forEach((row, idx) => {
        rankByModel.set(row.model_name, idx + 1);
      });
      maps[col.key] = rankByModel;
    });

  return maps;
}

function renderModelTable() {
  const wrap = document.getElementById("modelTableWrap");
  const searchEl = document.getElementById("searchModels");
  const familyEl = document.getElementById("familyFilter");
  const setupEl = document.getElementById("setupFilter");
  const sortEl = document.getElementById("sortMetric");
  if (!wrap || !searchEl || !familyEl || !setupEl || !sortEl) return;

  const keyword = searchEl.value.trim().toLowerCase();
  const family = familyEl.value;
  const setup = setupEl.value;
  const sortMetric = sortEl.value;

  const filtered = modelData.filter((row) => {
    const byName = row.model_name.toLowerCase().includes(keyword);
    const byFamily = family === "all" ? true : getFamilyName(row.model_name) === family;
    const bySetup = setup === "all" ? true : row.model_name.includes(setup);
    return byName && byFamily && bySetup;
  });

  const sorted = sortModels(filtered, sortMetric);

  const topRanks = getTopRankMaps(filtered, 3);

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");

  modelTableColumns.forEach((col) => {
    const th = document.createElement("th");

    if (!col.goal) {
      th.textContent = col.label;
      trHead.appendChild(th);
      return;
    }

    const headWrap = document.createElement("span");
    headWrap.className = "metric-head";

    const label = document.createElement("span");
    label.textContent = col.label;

    const goal = document.createElement("span");
    goal.className = `metric-goal ${col.goal === "max" ? "up" : "down"}`;
    goal.textContent = col.goal === "max" ? "↑" : "↓";
    goal.title = col.goal === "max" ? "Cang cao cang tot" : "Cang thap cang tot";
    goal.setAttribute("aria-label", goal.title);

    headWrap.appendChild(label);
    headWrap.appendChild(goal);
    th.appendChild(headWrap);
    trHead.appendChild(th);
  });

  thead.appendChild(trHead);

  const tbody = document.createElement("tbody");
  sorted.forEach((row) => {
    const tr = document.createElement("tr");

    modelTableColumns.forEach((col) => {
      const td = document.createElement("td");
      td.textContent = col.formatter(row[col.key]);

      if (col.goal) {
        const rank = topRanks[col.key].get(row.model_name);
        if (rank) {
          td.classList.add("metric-rank", `metric-rank-${rank}`);
          td.title = `Top ${rank} cho ${col.label}`;
        }
      }

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);

  wrap.innerHTML = "";
  wrap.appendChild(table);
}

function renderTopBars(metric = "macro_f1", topN = 8) {
  const wrap = document.getElementById("topBars");
  if (!wrap) return;

  const sorted = sortModels(modelData, metric).slice(0, topN);
  const maxVal = Math.max(...sorted.map((x) => x[metric]));

  wrap.innerHTML = "";
  sorted.forEach((row) => {
    const width = maxVal === 0 ? 0 : (row[metric] / maxVal) * 100;
    const bar = document.createElement("div");
    bar.className = "bar-row";
    bar.innerHTML = `
      <div class="bar-label">${row.model_name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${width.toFixed(2)}%"></div></div>
      <div class="bar-value">${fmt(row[metric])}</div>
    `;
    wrap.appendChild(bar);
  });
}

function renderEnsembleTable() {
  const wrap = document.getElementById("ensembleAlphaTable");
  if (!wrap) return;

  const columns = [
    { key: "alpha", label: "alpha", formatter: (v) => fmt(v, 1) },
    { key: "accuracy", label: "accuracy", goal: "max", formatter: (v) => fmt(v) },
    { key: "macro_f1", label: "macro_f1", goal: "max", formatter: (v) => fmt(v) },
    { key: "weighted_f1", label: "weighted_f1", goal: "max", formatter: (v) => fmt(v) },
    { key: "precision", label: "precision", goal: "max", formatter: (v) => fmt(v) },
    { key: "recall", label: "recall", goal: "max", formatter: (v) => fmt(v) },
    { key: "mae", label: "mae", goal: "min", formatter: (v) => fmt(v) }
  ];

  wrap.innerHTML = "";
  wrap.appendChild(createRankedMetricTable(columns, ensembleAlphaData));
}

function renderRobustnessTable() {
  const wrap = document.getElementById("robustnessTable");
  if (!wrap) return;

  const columns = [
    { key: "model_name", label: "model" },
    { key: "clean_accuracy", label: "clean_acc", goal: "max", formatter: (v) => fmt(v) },
    { key: "noisy_accuracy", label: "noisy_acc", goal: "max", formatter: (v) => fmt(v) },
    { key: "drop_accuracy", label: "drop_acc", goal: "min", formatter: (v) => fmt(v), dividerAfter: true },
    { key: "clean_macro_f1", label: "clean_macro_f1", goal: "max", formatter: (v) => fmt(v) },
    { key: "noisy_macro_f1", label: "noisy_macro_f1", goal: "max", formatter: (v) => fmt(v) },
    { key: "drop_macro_f1", label: "drop_macro_f1", goal: "min", formatter: (v) => fmt(v), dividerAfter: true },
    { key: "clean_mae", label: "clean_mae", goal: "min", formatter: (v) => fmt(v) },
    { key: "noisy_mae", label: "noisy_mae", goal: "min", formatter: (v) => fmt(v) },
    { key: "drop_mae", label: "drop_mae", goal: "min", formatter: (v) => fmt(v) }
  ];

  wrap.innerHTML = "";
  wrap.appendChild(createRankedMetricTable(columns, robustnessData));
}

function renderErrorSummary() {
  const wrap = document.getElementById("errorSummaryTable");
  if (!wrap) return;
  const headers = ["model_family", "checkpoint", "accuracy", "mae", "error_rate"];
  const rows = errorSummaryData.map((r) => [
    r.model_family,
    r.checkpoint,
    fmt(r.accuracy),
    fmt(r.mae),
    fmt(r.error_rate)
  ]);
  wrap.innerHTML = "";
  wrap.appendChild(createTable(headers, rows));
}

function renderErrorCategoryTable() {
  const wrap = document.getElementById("errorCategoryTable");
  if (!wrap) return;
  const headers = ["error_category", "count"];
  const rows = errorCategoryFocus.map((r) => [r.category, String(r.count)]);
  wrap.innerHTML = "";
  wrap.appendChild(createTable(headers, rows));
}

function renderCaseStudies() {
  const wrap = document.getElementById("caseStudyGrid");
  if (!wrap) return;

  wrap.innerHTML = "";
  caseStudies.forEach((item) => {
    const card = document.createElement("article");
    card.className = "case-card";
    const confText = item.confidence !== null ? ` | độ tin cậy=${fmt(item.confidence, 4)}` : "";
    card.innerHTML = `
      <p class="case-tag">${item.tag}</p>
      <h3>${item.title}</h3>
      <p class="case-meta">nhãn thật=${item.true_rating} | dự đoán=${item.pred_rating}${confText}</p>
      <p>${truncateText(item.review_text, 220)}</p>
      <p class="case-note">${item.note}</p>
      <p class="case-source">Source: ${item.source}</p>
      <button class="ghost-btn open-full-text" data-title="${item.title.replace(/"/g, "&quot;")}" data-body="${item.review_text.replace(/"/g, "&quot;")}">Xem toàn văn</button>
    `;
    wrap.appendChild(card);
  });
}

async function renderBertIgCases() {
  const wrap = document.getElementById("bertIgCaseGrid");
  if (!wrap) return;

  wrap.innerHTML = "";
  const rawMap = await loadBertIgRawMap();

  bertIgCases.forEach((item) => {
    const card = document.createElement("article");
    const ok = item.status === "correct";
    card.className = `ig-case-card ${ok ? "is-correct" : "is-incorrect"}`;

    const raw = rawMap.get(item.id);
    const lineHtml = raw?.lineHtml || renderIgLine(item.text_line, item.top_tokens);
    const topFallback = `<p class="ig-top-line"><em>Top tokens: ${item.top_tokens
      .map((t) => `${t.token}(${fmt(t.weight, 3)})`)
      .join(", ")}</em></p>`;
    const topTokensHtml = raw?.topTokensHtml || topFallback;

    card.innerHTML = `
      <div class="ig-case-head">
        <p class="ig-case-tag ${ok ? "ok" : "err"}">${ok ? "Case đúng" : "Case sai"}</p>
        <p class="ig-case-id">ID ${item.id} • ${item.source}</p>
      </div>
      <h4>${item.title}</h4>
      <p class="ig-case-metrics">True=${item.true_rating} | Pred=${item.pred_rating} | Confidence=${fmt(item.confidence, 4)}</p>
      <div class="ig-inline-line">${lineHtml}</div>
      ${topTokensHtml}
      <p class="ig-case-insight"><strong>IG insight:</strong> ${item.insight}</p>
    `;

    wrap.appendChild(card);
  });
}

function getGroupFilteredConfusions(group) {
  if (group === "weighted") return confusionImages.filter((m) => m.includes("weighted_ce"));
  if (group === "undersample") return confusionImages.filter((m) => m.includes("undersample_ce"));
  if (group === "bert") return confusionImages.filter((m) => m.startsWith("bert_"));
  if (group === "distilbert") return confusionImages.filter((m) => m.startsWith("distilbert_"));
  if (group === "rnn") return confusionImages.filter((m) => m.startsWith("bilstm"));
  return confusionImages;
}

function renderConfusionGallery() {
  const wrap = document.getElementById("confusionGallery");
  const groupEl = document.getElementById("cmGroup");
  if (!wrap || !groupEl) return;

  const models = getGroupFilteredConfusions(groupEl.value);
  wrap.innerHTML = "";

  models.forEach((name) => {
    const fig = document.createElement("figure");
    fig.className = "figure-card";
    fig.innerHTML = `
      <img src="../outputs/figures/confusion_matrix/${name}_confusion_matrix.png" alt="${name} confusion matrix" loading="lazy">
      <figcaption>${name}_confusion_matrix.png</figcaption>
    `;
    wrap.appendChild(fig);
  });

  bindImageLightbox();
}

function bindTabControls() {
  const buttons = [...document.querySelectorAll(".tab-btn")];
  const panels = [...document.querySelectorAll(".tab-panel")];
  if (!buttons.length || !panels.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
      panels.forEach((p) => p.classList.toggle("active", p.dataset.tabPanel === tab));
    });
  });
}

function renderDemoSelect() {
  const select = document.getElementById("demoCaseSelect");
  if (!select) return;

  select.innerHTML = "";
  demoCases.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label;
    select.appendChild(opt);
  });
}

function fillDemoFromSelection() {
  const select = document.getElementById("demoCaseSelect");
  const titleInput = document.getElementById("demoTitleInput");
  const reviewInput = document.getElementById("demoReviewInput");
  if (!select || !titleInput || !reviewInput) return;

  const found = demoCases.find((c) => c.id === select.value) || demoCases[0];
  if (!found) return;
  titleInput.value = found.title;
  reviewInput.value = found.review_text;
}

function resolveDemoResult(title, review, selectedId) {
  const selected = demoCases.find((c) => c.id === selectedId);
  const inputText = `${title || ""} ${review || ""}`.trim();

  if (selected) {
    const selectedText = `${selected.title} ${selected.review_text}`.trim();
    if (selectedText.toLowerCase() === inputText.toLowerCase()) {
      return { ...selected, mode: "exact" };
    }
  }

  let best = null;
  let bestScore = -1;
  demoCases.forEach((c) => {
    const score = tokenOverlapScore(inputText, `${c.title} ${c.review_text}`);
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  });

  return {
    ...best,
    mode: "nearest",
    overlap: bestScore
  };
}

function renderDemoResult(result) {
  const wrap = document.getElementById("demoResult");
  if (!wrap || !result) return;

  const verdict = result.pred_rating === result.true_rating ? "Đúng nhãn" : "Lệch nhãn";
  const confLine = result.confidence !== null ? `Độ tin cậy: ${fmt(result.confidence, 4)}` : "Độ tin cậy: n/a";
  const modeLine = result.mode === "nearest"
    ? `Chế độ: nearest cached case (token overlap=${fmt(result.overlap, 3)})`
    : "Chế độ: exact cached case";

  wrap.innerHTML = `
    <h3>Kết quả</h3>
    <p><strong>Rating dự đoán:</strong> ${result.pred_rating}</p>
    <p><strong>Rating tham chiếu:</strong> ${result.true_rating}</p>
    <p><strong>Trạng thái:</strong> ${verdict}</p>
    <p><strong>${confLine}</strong></p>
    <p>${modeLine}</p>
    <p><strong>Source:</strong> ${result.source}</p>
  `;
}

function bindHeaderCompact() {
  const header = document.querySelector(".site-header");
  if (!header) return;

  const ENTER_COMPACT_Y = 132;
  const EXIT_COMPACT_Y = 84;

  let isCompact = header.classList.contains("compact");
  let ticking = false;

  const apply = () => {
    const y = window.scrollY || window.pageYOffset || 0;

    if (!isCompact && y >= ENTER_COMPACT_Y) {
      isCompact = true;
      header.classList.add("compact");
    } else if (isCompact && y <= EXIT_COMPACT_Y) {
      isCompact = false;
      header.classList.remove("compact");
    }

    ticking = false;
  };

  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(apply);
  };

  apply();
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
}

function bindDemoForm() {
  const form = document.getElementById("demoForm");
  const select = document.getElementById("demoCaseSelect");
  const titleInput = document.getElementById("demoTitleInput");
  const reviewInput = document.getElementById("demoReviewInput");
  if (!form || !select || !titleInput || !reviewInput) return;

  select.addEventListener("change", fillDemoFromSelection);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const title = titleInput.value.trim();
    const review = reviewInput.value.trim();
    if (!review) return;

    const result = resolveDemoResult(title, review, select.value);
    renderDemoResult(result);
  });
}

function bindImageLightbox() {
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightboxImage");
  const closeBtn = document.getElementById("lightboxClose");
  if (!lightbox || !lightboxImg || !closeBtn) return;

  document.querySelectorAll(".figure-card img").forEach((img) => {
    img.addEventListener("click", () => {
      lightboxImg.src = img.src;
      lightbox.classList.add("show");
      lightbox.setAttribute("aria-hidden", "false");
    });
  });

  const close = () => {
    lightbox.classList.remove("show");
    lightbox.setAttribute("aria-hidden", "true");
    lightboxImg.src = "";
  };

  closeBtn.onclick = close;
  lightbox.onclick = (event) => {
    if (event.target === lightbox) close();
  };
}

function bindTextModal() {
  const modal = document.getElementById("textModal");
  const titleEl = document.getElementById("textModalTitle");
  const bodyEl = document.getElementById("textModalBody");
  const closeBtn = document.getElementById("textModalClose");
  if (!modal || !titleEl || !bodyEl || !closeBtn) return;

  const close = () => {
    modal.classList.remove("show");
    modal.setAttribute("aria-hidden", "true");
  };

  document.querySelectorAll(".open-full-text").forEach((btn) => {
    btn.addEventListener("click", () => {
      titleEl.textContent = btn.dataset.title || "Toàn văn";
      bodyEl.textContent = btn.dataset.body || "";
      modal.classList.add("show");
      modal.setAttribute("aria-hidden", "false");
    });
  });

  closeBtn.onclick = close;
  modal.onclick = (event) => {
    if (event.target === modal) close();
  };
}

function bindControls() {
  const searchEl = document.getElementById("searchModels");
  const familyEl = document.getElementById("familyFilter");
  const setupEl = document.getElementById("setupFilter");
  const sortEl = document.getElementById("sortMetric");
  const cmGroup = document.getElementById("cmGroup");

  [searchEl, familyEl, setupEl, sortEl].forEach((el) => {
    if (!el) return;
    el.addEventListener("input", renderModelTable);
    el.addEventListener("change", renderModelTable);
  });

  if (cmGroup) {
    cmGroup.addEventListener("change", renderConfusionGallery);
  }
}

function bindSectionObserver() {
  const links = [...document.querySelectorAll(".top-nav a")];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if (!sections.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const id = `#${entry.target.id}`;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === id));
      });
    },
    { rootMargin: "-35% 0px -50% 0px", threshold: 0.01 }
  );

  sections.forEach((section) => observer.observe(section));
}

function init() {
  renderHeroMeta();
  renderHeroHighlights();
  renderKpis();
  renderProblemStats();
  renderSampleCards();
  renderSplitTable();
  renderArchitectureFlow();

  renderCurveGallery("lossCurveGallery");
  renderCurveGallery("scoreCurveGallery");
  renderLrConfig();

  renderModelTable();
  renderTopBars("macro_f1", 10);
  renderEnsembleTable();
  renderRobustnessTable();
  renderConfusionGallery();

  renderBertIgCases();

  renderDemoSelect();
  fillDemoFromSelection();
  bindDemoForm();

  bindControls();
  bindTabControls();
  bindImageLightbox();
  bindTextModal();
  bindSectionObserver();
  bindHeaderCompact();
}

document.addEventListener("DOMContentLoaded", init);
