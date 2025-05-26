export function AboutPage({ onBack }: { onBack: () => void }) {
  return (
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <h2 style={{ marginBottom: '0.5rem', fontSize: '1.75rem' }}>Transformers: Gradients and Graphs</h2>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Motivation</h3>
        <p style={paragraphStyle}>
          Transformer models are widely used in modern NLP, but their internal reasoning remains hard to interpret. This tool provides visualizations that help reveal how attention is distributed across tokens and how much influence each token has on the model’s internal representations. By comparing attention weights and attention gradients with respect to input embeddings, we offer a clearer view of what the model focuses on — and why.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>How this tool works</h3>
        <div style={paragraphStyle}>
          Given a user-provided sentence, the tool processes it using a selected transformer model and task. For a specific attention layer, we compute:
          <ul style={{ paddingLeft: '1.2rem' }}>
            <li><strong>Attention matrix</strong>: shows how much each token attends to others.</li>
            <li><strong>Gradient matrix</strong>: shows the gradient norm of each attention score with respect to input embeddings, indicating token influence.</li>
          </ul>
          These matrices are rendered as interactive bipartite graphs to explore token-level dependencies.
        </div>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Overarching Goal</h3>
        <p style={paragraphStyle}>
          This project investigates whether attention weights and gradients can be meaningful proxies for model interpretability. From simple experimentation, we can already see that these aspects differ and activate at different layers.
          Additionally, results depend on the specific task, model architecture, and level of fine-tuning. </p>
          </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Tasks</h3>
        <p style={paragraphStyle}>
          While we try to span a wider range of tasks, it is a bit unscalable to fully integrate them. Therefore, we pick a smattering that spans task complexity. 
        </p>
        <ul style={{ paddingLeft: '1.2rem', ...paragraphStyle }}>
          <li><strong>MLM</strong>: Masked language modeling (e.g. BERT).</li>
          <li><strong>SST</strong>: Sentence-level sentiment classification.</li>
          <li><strong>MNLI</strong>: Natural language inference using premise-hypothesis pairs.</li>
        </ul>
      </div>
<div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Models</h3>
    The following models are used across tasks.
      <ul style={{ paddingLeft: '1.2rem', ...paragraphStyle }}>
          <li><strong>BERT</strong>:
            <ul style={{ paddingLeft: '1rem' }}>
              <li><code>bert-base-uncased</code> – base model used for MLM (no fine-tuning).</li>
              <li><code>textattack_bert-base-uncased-SST-2</code> – fully fine-tuned on SST-2.</li>
              <li><code>textattack_bert-base-uncased-MNLI</code> – fully fine-tuned on MNLI.</li>
            </ul>
          </li>
          <li><strong>DistilBERT</strong>:
            <ul style={{ paddingLeft: '1rem' }}>
              <li><code>distilbert-base-uncased</code> – base model used for MLM (no fine-tuning).</li>
              <li><code>distilbert-base-uncased-finetuned-sst-2-english</code> – fully fine-tuned on SST-2.</li>
              <li><code>textattack_distilbert-base-uncased-MNLI</code> – fully fine-tuned on MNLI.</li>
            </ul>
          </li>
          <li><strong>RoBERTa</strong>:
            <ul style={{ paddingLeft: '1rem' }}>
              <li><code>roberta-base</code> – base model used for MLM (no fine-tuning).</li>
              <li><code>textattack_roberta-base-SST-2</code> – fully fine-tuned on SST-2.</li>
              <li><code>roberta-large-mnli</code> – fully fine-tuned on MNLI.</li>
            </ul>
          </li>
        </ul>
      </div>


       <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Limitations</h3>
        <p style={paragraphStyle}>
          While we aim to compare architectures, pretraining strategies, fine-tuning methods, and task complexity, we currently rely on models publicly available on Hugging Face. This limits our ability to ensure apples-to-apples comparisons. For example, MNLI is available for RoBERTa only in the large version, not the base.
        </p>
        <p style={paragraphStyle}>
          Additionally, gradient computations are performed server-side and are computationally expensive. We initially deployed using a T4 GPU, but the tool is now running on a more affordable CPU-only backend. This causes noticeable latency, especially for longer sequences or higher layer counts.
        </p>
        <p style={paragraphStyle}>
          If there is sustained interest in the tool, we hope to obtain support to re-enable GPU-backed serving and expand model coverage in a consistent, reproducible way.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>About me</h3>
        <p style={paragraphStyle}>
          I'm <a href="https://optimalyifansun.com/" target="_blank" rel="noopener noreferrer">Yifan Sun</a>, from Stony Brook University. My goal in creating this app — and other <a href="https://www.optimalvisualizer.com/" target="_blank" rel="noopener noreferrer">interactive visual tools</a> — is to help bridge the gap between abstract theory and intuitive understanding in machine learning. Tools like this one aim to make internal model mechanics more accessible to students, researchers, and practitioners.
        </p>
      </div>

      <button onClick={onBack} style={{ alignSelf: 'flex-start', marginTop: '1rem' }}>← Back to main</button>
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  backgroundColor: '#f9f9f9',
  border: '1px solid #ddd',
  borderRadius: '8px',
  padding: '1rem',
  boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '1.1rem',
  fontWeight: 600,
  marginBottom: '0.5rem'
};

const paragraphStyle: React.CSSProperties = {
  fontSize: '0.95rem',
  lineHeight: 1.6
};