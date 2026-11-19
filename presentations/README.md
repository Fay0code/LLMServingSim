# Research presentations

[Download the FFN pruning proposal](ffn-pruning-patent-proposal.pptx).
On GitHub, open the file and select **Download raw file** to save the editable PowerPoint.

The Chinese-language, three-slide deck covers limitations of existing pruning
strategies, a baseline parameter-count lower bound, and profiling-driven FFN
width selection. It is a proposed research design, not a supported simulator
feature or a report of validated accuracy or speedup. Parameter count is a
capacity proxy, not an accuracy guarantee. Patent claims require prior-art
review and professional review.

## Figures

- [Pruning feedback workflow](ffn-pruning-feedback-workflow.png): baseline pruning,
  parameter-count lower bound, inference-latency simulation and channel feedback.
- [Before-and-after comparison](ffn-pruning-before-after.png): FFN channel
  structure and illustrative hardware execution blocks.

These standalone proposal figures are unrelated to the simulator implementation.
Hardware blocks illustrate a hypothetical mechanism, not measured performance.
