---
layout: single
title: "Code"
permalink: /code/
author_profile: true
---  


---
layout: single
title: "Code"
permalink: /code/
author_profile: true
---

This page provides the PyTorch implementation and executed Jupyter Notebook for the study **Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST**.

The original notebook contains the experimental implementation and recorded results. The HTML export provides a browser-readable version with code, explanations, saved outputs, tables, and figures.

---

## Jupyter Notebook — Executable Source

The original `.ipynb` file contains the model architectures, training and validation procedures, layer-wise gradient measurements, experimental comparisons, and analysis code.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb"
     download>
    ⬇ Download .ipynb
  </a>

  <a class="code-btn code-secondary-btn"
     href="https://github.com/uyentran9/gradient-vanishing/blob/main/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb"
     target="_blank"
     rel="noopener noreferrer">
    ↗ View on GitHub
  </a>
</div>

---

## Executed Notebook — HTML Preview

The HTML export preserves the notebook's saved code cells, Markdown explanations, experimental outputs, tables, and visualizations without requiring the experiments to be rerun.

<div class="code-actions">
  <a class="code-btn code-primary-btn"
     href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html"
     download>
    ⬇ Download HTML
  </a>

  <a class="code-btn code-secondary-btn"
     href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html"
     target="_blank"
     rel="noopener noreferrer">
    ↗ Open full-screen
  </a>
</div>

<div class="notebook-wrap">
  <iframe
    src="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html"
    title="Gradient Vanishing — Executed Jupyter Notebook"
    loading="lazy">
  </iframe>
</div>

<style>
/* Button layout */
.code-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75em;
  margin: 1em 0 1.5em;
}

/* Shared button styling */
.code-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 4px;
  text-decoration: none !important;
  transition: opacity 0.2s ease;
}

/* Primary download button */
.code-primary-btn,
.code-primary-btn:visited {
  background: #287f92 !important;
  color: #ffffff !important;
  border: 1px solid #287f92 !important;
}

/* Secondary view button */
.code-secondary-btn,
.code-secondary-btn:visited {
  background: transparent !important;
  color: #287f92 !important;
  border: 1px solid #287f92 !important;
}

/* Hover effect */
.code-btn:hover {
  opacity: 0.85;
  text-decoration: none !important;
}

/* Notebook preview container */
.notebook-wrap {
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  background: #ffffff;
}

/* Embedded HTML notebook */
.notebook-wrap iframe {
  display: block;
  width: 100%;
  height: 85vh;
  min-height: 700px;
  border: 0;
}
</style>


<!--  

This page provides the PyTorch implementation and executed Jupyter Notebook for the study **Investigating Vanishing Gradients and Early-Layer Trainability Across Network Depth in Deep MLPs on BloodMNIST**.

The source notebook supports inspection and reproduction of the experiments. The HTML export provides a read-only version with saved code cells, explanations, outputs, tables, and figures.

---  

## Jupyter Notebook — Executable Source  

The original `.ipynb` file contains the experimental implementation, including model architectures, training procedures, gradient measurements, and analysis.  

<p>This page embeds the fully executed notebook, exported directly from Jupyter — every cell, output, and chart exactly as run.</p>

<div class="code-actions">
  <a class="data-download-btn" href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb" download>⬇ Download .ipynb</a>
  <a class="data-download-btn code-secondary-btn" href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb" target="_blank" rel="noopener">↗ Open full-screen</a>
</div>

<div id="notebook-wrap" class="notebook-wrap">
  <iframe id="notebook-frame" src="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb" loading="lazy"></iframe>
</div>

---

## Executed Notebook — HTML Preview  

The HTML export presents the notebook in a browser-readable format, preserving its saved experimental outputs and visualizations without requiring the notebook to be executed.  

<div class="code-actions">
  <a class="data-download-btn" href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html" download>⬇ Download HTML</a>
  <a class="data-download-btn code-secondary-btn" href="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html" target="_blank" rel="noopener">↗ Open full-screen</a>
</div>

<div id="notebook-wrap" class="notebook-wrap">
  <iframe id="notebook-frame" src="{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.html" loading="lazy"></iframe>
</div>

<style>
.code-actions { margin: 1em 0 1.5em; display: flex; gap: 0.75em; flex-wrap: wrap; }
.data-download-btn {
  display: inline-block;
  padding: 0.5em 1.1em;
  font-size: 0.9rem;
  font-weight: 600;
  color: #fff !important;
  background: #287f92;
  border-radius: 4px;
  text-decoration: none !important;
}
.data-download-btn:hover { opacity: 0.85; }
.code-secondary-btn,
.code-secondary-btn:visited {
  background: transparent !important;
  color: #287f92 !important;
  border: 1px solid #287f92 !important;
}
.notebook-wrap {
  border: 1px solid var(--global-border-color, #e8e8e8);
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  overflow: hidden;
  background: #fff;
}
.notebook-wrap iframe {
  width: 100%;
  height: 85vh;
  min-height: 700px;
  border: 0;
  display: block;
}
.notebook-missing {
  padding: 3em 1.5em;
  text-align: center;
  color: var(--global-text-color-light, #888);
}
.notebook-missing code {
  background: var(--global-code-background-color, #f7f7f8);
  padding: 0.15em 0.4em;
  border-radius: 3px;
}
</style>

<script>
(function () {
  var url = '{{ site.baseurl }}/files/gradient_vanishing_bloodmnist_uyentran_v4.ipynb';
  fetch(url, { method: 'HEAD' }).then(function (res) {
    if (!res.ok) throw new Error('missing');
  }).catch(function () {
    document.getElementById('notebook-wrap').innerHTML =
      '<div class="notebook-missing">The notebook export has not been uploaded yet.<br>' +
      'Expected at <code>' + url + '</code>.</div>';
  });
})();
</script>
-->  


<p style="margin-top:24px;">
<a href="{{ '/' | relative_url }}" style="background:#1f4d7a;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;"> 🏠 Back to Home </a>
</p>

Thank you for visiting — and for being part of my journey 🌱  
