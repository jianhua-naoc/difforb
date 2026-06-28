function renderKatexMath() {
  if (typeof renderMathInElement !== "function") {
    return;
  }

  const options = {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    throwOnError: false,
    strict: "ignore",
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"]
  };

  document.querySelectorAll(".arithmatex").forEach((element) => {
    if (element.dataset.katexRendered === "true") {
      return;
    }
    renderMathInElement(element, options);
    element.dataset.katexRendered = "true";
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    renderKatexMath();
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    renderKatexMath();
  });
}
