const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        AlignmentType, WidthType, ShadingType, BorderStyle, HeadingLevel, PageBreak, LevelFormat } = require('docx');
const fs = require('fs');

const border = { style: BorderStyle.SINGLE, size: 4, color: "2E75B6" };
const borders = { top: border, bottom: border, left: border, right: border };

const doc = new Document({
  numbering: {
    config: [
      { reference: "numbers",
        levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        ] 
      },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "1F4E78" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "5B9BD5" },
        paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [
      // Title Page
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        alignment: AlignmentType.CENTER,
        children: [new TextRun("VisionGuard SDK")]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 240 },
        children: [new TextRun({ text: "Technical Documentation", bold: true, size: 28 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 480 },
        children: [new TextRun({ text: "Automated Drift Detection, Local Adaptation,", size: 24 })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "and Federated Learning for Computer Vision Models", size: 24 })]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "" }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Version 1.0", size: 20, italics: true })]
      }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Table of Contents
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("Table of Contents")]
      }),
      new Paragraph({ text: "1. System Overview" }),
      new Paragraph({ text: "2. Layer 1: Client CV Model" }),
      new Paragraph({ text: "3. Layer 2: SDK Wrapper Layer" }),
      new Paragraph({ text: "4. Layer 3: Drift Detection Layer" }),
      new Paragraph({ text: "5. Layer 4: Decision & Reliability Layer" }),
      new Paragraph({ text: "6. Layer 5: Federated Learning Server" }),
      new Paragraph({ text: "7. Algorithms & Formulas" }),
      new Paragraph({ text: "8. Key Parameters Reference" }),
      new Paragraph({ text: "9. Implementation Guide" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 1: System Overview
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("1. System Overview")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({
        text: "VisionGuard SDK is an automated framework that wraps existing computer vision models to provide real-time drift detection, local model adaptation, and federated learning capabilities without requiring model architecture changes."
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("1.1 Key Features")]
      }),
      new Paragraph({ text: "• Automatic drift detection using three complementary metrics" }),
      new Paragraph({ text: "• Local adaptation triggered by reliability thresholds" }),
      new Paragraph({ text: "• Offline/online mode with automatic synchronization" }),
      new Paragraph({ text: "• Real-time monitoring dashboard" }),
      new Paragraph({ text: "• Privacy-preserving federated learning integration" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("1.2 Architecture Layers")]
      }),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2340, 3510, 3510],
        rows: [
          new TableRow({
            children: ["Layer", "Purpose", "Output"].map(text => 
              new TableCell({
                borders,
                width: { size: text === "Layer" ? 2340 : 3510, type: WidthType.DXA },
                shading: { fill: "2E75B6", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF" })] })]
              })
            )
          }),
          ...[
            ["Layer 1", "Client CV Model", "Predictions, confidence, embeddings"],
            ["Layer 2", "SDK Wrapper", "Intercepted runtime metrics"],
            ["Layer 3", "Drift Detection", "Drift score (0-1)"],
            ["Layer 4", "Decision & Reliability", "Reliability score, adaptation trigger"],
            ["Layer 5", "FL Server", "Global model updates"]
          ].map(row => new TableRow({
            children: row.map((text, i) => new TableCell({
              borders,
              width: { size: i === 0 ? 2340 : 3510, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ text })]
            }))
          }))
        ]
      }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 2: Client CV Model
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("2. Layer 1: Client CV Model")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.1 Purpose")]
      }),
      new Paragraph({
        text: "The client CV model is the existing computer vision model deployed by the user. VisionGuard SDK is model-agnostic and works with any PyTorch-based CV model."
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.2 Supported Model Types")]
      }),
      new Paragraph({ text: "• Image Classification (ResNet, EfficientNet, ViT)" }),
      new Paragraph({ text: "• Object Detection (YOLO, Faster R-CNN, RetinaNet)" }),
      new Paragraph({ text: "• Segmentation (UNet, DeepLab, Mask R-CNN)" }),
      new Paragraph({ text: "• Face Recognition (FaceNet, ArcFace)" }),
      new Paragraph({ text: "• Medical Imaging (Custom CNNs)" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.3 Required Model Properties")]
      }),
      new Paragraph({ text: "The model must provide:" }),
      new Paragraph({ text: "  1. Forward pass returning logits or predictions" }),
      new Paragraph({ text: "  2. Access to intermediate embeddings (feature vectors)" }),
      new Paragraph({ text: "  3. PyTorch nn.Module implementation" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("2.4 Integration Example")]
      }),
      new Paragraph({ text: "from visionguard import VisionGuardSDK" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Your existing model" }),
      new Paragraph({ text: "model = YourCVModel()" }),
      new Paragraph({ text: "model.load_state_dict(torch.load('model.pth'))" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Wrap with VisionGuard" }),
      new Paragraph({ text: "sdk = VisionGuardSDK(model, device='cuda')" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Use as normal" }),
      new Paragraph({ text: "result = sdk.predict(image_tensor)" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 3: SDK Wrapper Layer
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("3. Layer 2: SDK Wrapper Layer")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.1 Purpose")]
      }),
      new Paragraph({
        text: "The SDK wrapper intercepts model outputs without modifying the model architecture. It uses PyTorch forward hooks to extract runtime metrics automatically."
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.2 Intercepted Metrics")]
      }),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2800, 3280, 3280],
        rows: [
          new TableRow({
            children: ["Metric", "Extraction Method", "Purpose"].map(text => 
              new TableCell({
                borders,
                width: { size: text === "Metric" ? 2800 : 3280, type: WidthType.DXA },
                shading: { fill: "5B9BD5", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF" })] })]
              })
            )
          }),
          ...[
            ["Predictions", "Softmax(logits)", "Final classification output"],
            ["Confidence", "max(softmax)", "Model certainty"],
            ["Embeddings", "Pre-classifier layer", "Internal representations"]
          ].map(row => new TableRow({
            children: row.map((text, i) => new TableCell({
              borders,
              width: { size: i === 0 ? 2800 : 3280, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ text })]
            }))
          }))
        ]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("3.3 Key Parameters")]
      }),
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2800, 2800, 3760],
        rows: [
          new TableRow({
            children: ["Parameter", "Default", "Description"].map(text => 
              new TableCell({
                borders,
                width: { size: text === "Description" ? 3760 : 2800, type: WidthType.DXA },
                shading: { fill: "D9E9F7", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text, bold: true })] })]
              })
            )
          }),
          ...[
            ["window_size", "500", "Sliding window for drift detection"],
            ["drift_threshold", "0.3", "Threshold to trigger adaptation (0-1)"],
            ["reliability_threshold", "0.7", "Min reliability for FL participation"],
            ["auto_adapt", "True", "Enable automatic local adaptation"],
            ["enable_ui", "True", "Enable monitoring dashboard"]
          ].map(row => new TableRow({
            children: row.map((text, i) => new TableCell({
              borders,
              width: { size: i === 2 ? 3760 : 2800, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ text })]
            }))
          }))
        ]
      }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 4: Drift Detection
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("4. Layer 3: Drift Detection Layer")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.1 Three-Metric Approach")]
      }),
      new Paragraph({
        text: "Drift detection combines three complementary signals to provide robust detection across different types of distribution shift."
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun("4.1.1 Metric 1: Prediction Confidence")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula:" }),
      new Paragraph({ text: "confidence = max(softmax(logits))" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Δconfidence = mean(baseline_confidence) - mean(current_confidence)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Normalized:" }),
      new Paragraph({ text: "conf_component = clip(Δconfidence / 0.3, 0, 1)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Interpretation:" }),
      new Paragraph({ text: "• High confidence drop → Model uncertain about predictions" }),
      new Paragraph({ text: "• Indicates covariate shift (input distribution changed)" }),
      new Paragraph({ text: "• Fast detection signal (immediate response)" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun("4.1.2 Metric 2: Output Entropy")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula:" }),
      new Paragraph({ text: "H(p) = -Σ pᵢ log(pᵢ)" }),
      new Paragraph({ text: "where p = softmax(logits)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Δentropy = mean(current_entropy) - mean(baseline_entropy)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Normalized:" }),
      new Paragraph({ text: "entropy_component = clip(Δentropy / 0.5, 0, 1)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Interpretation:" }),
      new Paragraph({ text: "• Increased entropy → Ambiguous decision boundaries" }),
      new Paragraph({ text: "• Model can't distinguish between classes clearly" }),
      new Paragraph({ text: "• Complements confidence metric" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun("4.1.3 Metric 3: Embedding Distribution Shift (CORE)")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula (KL Divergence for Gaussian distributions):" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "KL(P_baseline || P_current) = log(σ_current / σ_baseline) +" }),
      new Paragraph({ text: "    (σ_baseline² + (μ_baseline - μ_current)²) / (2σ_current²) - 0.5" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Where:" }),
      new Paragraph({ text: "• μ_baseline, σ_baseline = mean and std of baseline embeddings" }),
      new Paragraph({ text: "• μ_current, σ_current = mean and std of current window embeddings" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Normalized:" }),
      new Paragraph({ text: "embedding_component = clip(mean(KL) / 1.0, 0, 1)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Interpretation:" }),
      new Paragraph({ text: "• Most sensitive drift indicator" }),
      new Paragraph({ text: "• Captures internal representation changes" }),
      new Paragraph({ text: "• Detects semantic distribution shifts" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.2 Combined Drift Score")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula:" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "drift_score = w₁ × conf_component + w₂ × entropy_component + w₃ × embedding_component" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Default Weights:" }),
      new Paragraph({ text: "• w₁ = 0.3 (Confidence)" }),
      new Paragraph({ text: "• w₂ = 0.3 (Entropy)" }),
      new Paragraph({ text: "• w₃ = 0.4 (Embedding) — Highest weight" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Range: drift_score ∈ [0, 1]" }),
      new Paragraph({ text: "• 0.0 = No drift (perfect match to baseline)" }),
      new Paragraph({ text: "• 1.0 = Maximum drift (complete distribution shift)" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("4.3 Sliding Window Mechanism")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Window Parameters:" }),
      new Paragraph({ text: "• Window size: 500-1000 inferences (default: 500)" }),
      new Paragraph({ text: "• Update frequency: After each inference" }),
      new Paragraph({ text: "• Computation: Only when window full" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Benefits:" }),
      new Paragraph({ text: "• Smooths out noise from individual samples" }),
      new Paragraph({ text: "• Provides statistical significance" }),
      new Paragraph({ text: "• Adapts to gradual drift" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 5: Decision Layer
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("5. Layer 4: Decision & Reliability Layer")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("5.1 Reliability Score Computation")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula:" }),
      new Paragraph({ text: "reliability_score = 1 - drift_score" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Interpretation:" }),
      new Paragraph({ text: "• High reliability (>0.7) → Model predictions trustworthy" }),
      new Paragraph({ text: "• Low reliability (<0.7) → Model needs adaptation" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("5.2 Decision Flow")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({ text: "Decision 1: Drift Detection Alert" }),
      new Paragraph({ text: "IF drift_score > drift_threshold (0.3):" }),
      new Paragraph({ text: "    → Log drift event" }),
      new Paragraph({ text: "    → Alert user/dashboard" }),
      new Paragraph({ text: "    → Proceed to Decision 2" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({ text: "Decision 2: Local Adaptation Trigger" }),
      new Paragraph({ text: "IF reliability_score < reliability_threshold (0.7):" }),
      new Paragraph({ text: "    → Collect recent data samples" }),
      new Paragraph({ text: "    → Perform local fine-tuning" }),
      new Paragraph({ text: "    → Re-compute drift metrics" }),
      new Paragraph({ text: "    → Proceed to Decision 3" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({ text: "Decision 3: FL Participation" }),
      new Paragraph({ text: "IF reliability_score >= reliability_threshold AND is_online:" }),
      new Paragraph({ text: "    → Compute weight deltas: ΔW = W_adapted - W_original" }),
      new Paragraph({ text: "    → Send to FL server with reliability_score" }),
      new Paragraph({ text: "ELSE IF offline:" }),
      new Paragraph({ text: "    → Store updates in local queue" }),
      new Paragraph({ text: "    → Sync when connection restored" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("5.3 Local Adaptation Algorithm")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Algorithm: Fine-tuning with Adam optimizer" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Input:" }),
      new Paragraph({ text: "  • Current model state: W_current" }),
      new Paragraph({ text: "  • Recent data samples: D_recent (100-500 samples)" }),
      new Paragraph({ text: "  • Adaptation epochs: E (default: 3-5)" }),
      new Paragraph({ text: "  • Learning rate: α (default: 1e-4)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Process:" }),
      new Paragraph({ text: "  1. Save original weights: W_original ← W_current" }),
      new Paragraph({ text: "  2. FOR epoch in 1..E:" }),
      new Paragraph({ text: "       FOR batch in D_recent:" }),
      new Paragraph({ text: "         loss ← CrossEntropyLoss(model(x), y)" }),
      new Paragraph({ text: "         W_current ← W_current - α∇loss" }),
      new Paragraph({ text: "  3. Compute deltas: ΔW ← W_current - W_original" }),
      new Paragraph({ text: "  4. Re-establish baseline statistics" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Output:" }),
      new Paragraph({ text: "  • Adapted model: W_adapted = W_current" }),
      new Paragraph({ text: "  • Weight deltas: ΔW" }),
      new Paragraph({ text: "  • New reliability_score" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 6: FL Server
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("6. Layer 5: Federated Learning Server")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("6.1 Purpose")]
      }),
      new Paragraph({
        text: "The FL server aggregates model updates from multiple clients while preserving privacy. It never receives raw data, only weight deltas and metadata."
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("6.2 Privacy Guarantees")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Server NEVER receives:" }),
      new Paragraph({ text: "✗ Raw images or video frames" }),
      new Paragraph({ text: "✗ Embeddings or feature vectors" }),
      new Paragraph({ text: "✗ Labels or ground truth" }),
      new Paragraph({ text: "✗ Any personally identifiable information" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Server ONLY receives:" }),
      new Paragraph({ text: "✓ Weight deltas (ΔW)" }),
      new Paragraph({ text: "✓ Reliability scores" }),
      new Paragraph({ text: "✓ Drift signatures (statistical metadata)" }),
      new Paragraph({ text: "✓ Client ID (anonymized)" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("6.3 Reliability-Weighted Aggregation")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Formula (Modified FedAvg):" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Global_ΔW = Σᵢ (reliabilityᵢ × ΔWᵢ) / Σᵢ reliabilityᵢ" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Where:" }),
      new Paragraph({ text: "• i = client index" }),
      new Paragraph({ text: "• reliabilityᵢ = reliability score from client i" }),
      new Paragraph({ text: "• ΔWᵢ = weight deltas from client i" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "Benefits:" }),
      new Paragraph({ text: "• Stable clients (high reliability) dominate aggregation" }),
      new Paragraph({ text: "• Drifting clients (low reliability) are down-weighted" }),
      new Paragraph({ text: "• Prevents degraded models from polluting global model" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("6.4 Update Broadcast")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "After aggregation:" }),
      new Paragraph({ text: "  1. Server computes: W_global_new = W_global_old + Global_ΔW" }),
      new Paragraph({ text: "  2. Broadcast W_global_new to all connected clients" }),
      new Paragraph({ text: "  3. Clients evaluate improvement on validation set" }),
      new Paragraph({ text: "  4. Clients apply update if beneficial" }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 7: Algorithms Summary
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("7. Algorithms & Formulas Summary")]
      }),
      new Paragraph({ text: "" }),
      
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2340, 7020],
        rows: [
          new TableRow({
            children: ["Component", "Formula"].map(text => 
              new TableCell({
                borders,
                width: { size: text === "Component" ? 2340 : 7020, type: WidthType.DXA },
                shading: { fill: "FFE699", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text, bold: true })] })]
              })
            )
          }),
          ...[
            ["Confidence", "max(softmax(logits))"],
            ["Entropy", "H(p) = -Σ pᵢ log(pᵢ)"],
            ["KL Divergence", "log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 0.5"],
            ["Drift Score", "w₁×Δconf + w₂×Δent + w₃×KL"],
            ["Reliability", "1 - drift_score"],
            ["FL Aggregation", "Σ(reliabilityᵢ × ΔWᵢ) / Σreliabilityᵢ"]
          ].map(row => new TableRow({
            children: row.map((text, i) => new TableCell({
              borders,
              width: { size: i === 0 ? 2340 : 7020, type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ text })]
            }))
          }))
        ]
      }),
      
      new Paragraph({ text: "" }),
      new Paragraph({ text: "" }),
      
      // Chapter 8: Parameters
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("8. Key Parameters Reference")]
      }),
      new Paragraph({ text: "" }),
      
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        columnWidths: [2600, 1560, 1560, 3640],
        rows: [
          new TableRow({
            children: ["Parameter", "Default", "Range", "Impact"].map((text, i) => 
              new TableCell({
                borders,
                width: { size: [2600, 1560, 1560, 3640][i], type: WidthType.DXA },
                shading: { fill: "C6E0B4", type: ShadingType.CLEAR },
                margins: { top: 80, bottom: 80, left: 120, right: 120 },
                children: [new Paragraph({ children: [new TextRun({ text, bold: true })] })]
              })
            )
          }),
          ...[
            ["window_size", "500", "100-2000", "Larger = smoother but slower detection"],
            ["drift_threshold", "0.3", "0.1-0.5", "Lower = more sensitive to drift"],
            ["reliability_threshold", "0.7", "0.5-0.9", "Higher = stricter FL participation"],
            ["w_confidence", "0.3", "0-1", "Weight for confidence metric"],
            ["w_entropy", "0.3", "0-1", "Weight for entropy metric"],
            ["w_embedding", "0.4", "0-1", "Weight for embedding metric (should be highest)"],
            ["adapt_epochs", "5", "1-10", "More epochs = stronger adaptation"],
            ["adapt_lr", "1e-4", "1e-5 to 1e-3", "Learning rate for fine-tuning"]
          ].map(row => new TableRow({
            children: row.map((text, i) => new TableCell({
              borders,
              width: { size: [2600, 1560, 1560, 3640][i], type: WidthType.DXA },
              margins: { top: 80, bottom: 80, left: 120, right: 120 },
              children: [new Paragraph({ text })]
            }))
          }))
        ]
      }),
      
      new Paragraph({ children: [new PageBreak()] }),
      
      // Chapter 9: Implementation
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun("9. Implementation Guide")]
      }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("9.1 Installation")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "pip install visionguard-sdk" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("9.2 Basic Usage")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "from visionguard import VisionGuardSDK" }),
      new Paragraph({ text: "from your_model import YourCVModel" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Initialize" }),
      new Paragraph({ text: "model = YourCVModel()" }),
      new Paragraph({ text: "sdk = VisionGuardSDK(model, device='cuda')" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Establish baseline" }),
      new Paragraph({ text: "sdk.establish_baseline(clean_dataloader)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Use in production" }),
      new Paragraph({ text: "for image in production_stream:" }),
      new Paragraph({ text: "    result = sdk.predict(image)" }),
      new Paragraph({ text: "    # Drift detection automatic!" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("9.3 Monitoring Dashboard")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "from visionguard.ui import MonitoringDashboard" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "dashboard = MonitoringDashboard(sdk)" }),
      new Paragraph({ text: "dashboard.show()  # Real-time visualization" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("9.4 Offline Mode")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Initialize without FL server" }),
      new Paragraph({ text: "sdk = VisionGuardSDK(model, fl_server_url=None)" }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "# Updates stored locally" }),
      new Paragraph({ text: "# Later, when online:" }),
      new Paragraph({ text: "sdk.is_online = True" }),
      new Paragraph({ text: "sdk.sync_pending_updates()" }),
      new Paragraph({ text: "" }),
      
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun("9.5 Generated Outputs")]
      }),
      new Paragraph({ text: "" }),
      new Paragraph({ text: "The SDK automatically generates:" }),
      new Paragraph({ text: "• Real-time monitoring dashboard" }),
      new Paragraph({ text: "• Drift detection reports (PNG)" }),
      new Paragraph({ text: "• Adaptation cycle logs (JSON)" }),
      new Paragraph({ text: "• Model state checkpoints (PTH)" }),
      new Paragraph({ text: "• Performance metrics (JSON)" }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/mnt/user-data/outputs/VisionGuard_Technical_Documentation.docx", buffer);
  console.log("✓ Technical documentation created!");
});
