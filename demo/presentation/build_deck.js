// Builds the 5-slide internship deck: Skin Lesion Classifier (Melanoma vs Non-melanoma).
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("C:/Users/User/AppData/Roaming/Claude-Account2/local-agent-mode-sessions/skills-plugin/4dd9bccd-0e81-47a5-982f-2f246cdf1bf5/1893dce9-a7cc-4825-94c0-625d26af2b31/skills/pptx/scripts/apply_theme.js");

const OUT = "C:/Users/User/Desktop/RESEARCH_ITPU/demo/presentation/Skin_Lesion_Classifier_Demo.pptx";

// Same palette as the demo web page: teal accent, dark ink, melanoma red.
const THEME = {
  name: "Skin Lesion Classifier",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2327", lt1: "FFFFFF", dk2: "1F5F6B", lt2: "EEF4F5",
    accent1: "1F5F6B", accent2: "A3352B", accent3: "5F6B73", accent4: "2F6F4A",
    accent5: "D9E7E9", accent6: "F7E6E3", hlink: "1F5F6B", folHlink: "5F6B73",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- Layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.85, w: 6.1, h: 2.35, fontSize: 32, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.4, w: 6.1, h: 1.0, fontSize: 18, color: C.accent5, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 8.8, h: 0.75, fontSize: 30, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

const T = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: "top", ...o });
const card = (slide, x, y, w, h, fill, name) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
const badge = (slide, x, y, label, fill, textColor, name) => {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill }, line: { color: fill }, objectName: name });
  T(slide, label, { x, y, w: 0.42, h: 0.42, fontSize: 15, bold: true, color: textColor, align: "center", valign: "middle", objectName: name + " label" });
};

// ---------- 1. Title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection", { placeholder: "title" });
s.addText("Can dermoscopy-trained convolutional neural networks generalise to smartphone images?", { placeholder: "body" });
// Motif: concentric rings, like a lesion seen through a dermatoscope.
[[7.05, 1.15, 3.3, "1A5560"], [7.5, 1.6, 2.4, "2A6E7A"], [7.95, 2.05, 1.5, "3C808C"]].forEach(([x, y, d, col], i) =>
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: col }, line: { color: col }, objectName: `Lesion ring ${i + 1}` }));
s.addShape(pres.shapes.OVAL, { x: 8.37, y: 2.47, w: 0.66, h: 0.66, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "Lesion core" });
s.addNotes(
  "Hello everyone. My research is called Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection. " +
  "In simple words: I built an AI that looks at a close-up image of a skin spot and decides melanoma or not melanoma, " +
  "and I checked whether it still works, and whether its explanations still make sense, when the image type changes. " +
  "I'll take about three minutes to explain the idea, the data and the results, and then I'll show you the live demo."
);

// ---------- 2. Problem and goal ----------
pres.addSection({ title: "Project" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project" });
s.addText("Why this project", { placeholder: "title" });
card(s, 0.6, 1.4, 4.0, 3.3, C.background2, "Problem card");
T(s, "THE PROBLEM", { x: 0.95, y: 1.7, w: 3.3, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 1 });
T(s, "Melanoma is a dangerous skin cancer", { x: 0.95, y: 2.05, w: 3.3, h: 0.75, fontSize: 21, bold: true });
T(s, "Finding it early matters. Many skin spots look alike, so a second opinion from software could help doctors decide what to check first.", { x: 0.95, y: 2.95, w: 3.3, h: 1.5, fontSize: 15, color: C.accent3 });
s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.78, y: 2.8, w: 0.45, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "Arrow" });
card(s, 5.4, 1.4, 4.0, 3.3, C.text2, "Goal card");
T(s, "MY GOAL", { x: 5.75, y: 1.7, w: 3.3, h: 0.3, fontSize: 12, bold: true, color: C.accent5, charSpacing: 1 });
T(s, "An AI that answers one question", { x: 5.75, y: 2.05, w: 3.3, h: 0.75, fontSize: 21, bold: true, color: C.background1 });
T(s, [
  { text: "Melanoma or not?", options: { bullet: true, breakLine: true } },
  { text: "Where in the image did it look?", options: { bullet: true, breakLine: true } },
  { text: "Running in a simple web app", options: { bullet: true } },
], { x: 5.75, y: 2.95, w: 3.3, h: 1.5, fontSize: 15, color: C.background1, paraSpaceAfter: 6 });
s.addNotes(
  "Why this project? Melanoma is a dangerous type of skin cancer, and finding it early matters. " +
  "But many skin spots look very similar. " +
  "So my goal was an AI that answers one simple question: is this melanoma or not? " +
  "It also shows where in the image it looked, and it runs in a small web app that I'll show you."
);

// ---------- 3. Data and model ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Project" });
s.addText("How I built it", { placeholder: "title" });
const steps = [
  ["1", "Data", "HAM10000: 10,015 dermoscopy images with a diagnosis"],
  ["2", "Split", "6,938 to learn, 1,529 to tune, 1,548 kept for the final test"],
  ["3", "Model", "ResNet-50: learned from everyday photos first, then retrained for melanoma"],
  ["4", "Training", "8 rounds, about 9 minutes on a laptop GPU (RTX 5060)"],
];
steps.forEach(([n, head, body], i) => {
  const x = 0.6 + i * 2.25;
  card(s, x, 1.35, 2.05, 2.25, C.background2, `Step ${n} card`);
  badge(s, x + 0.22, 1.57, n, C.accent1, C.background1, `Step ${n} badge`);
  T(s, head, { x: x + 0.22, y: 2.12, w: 1.65, h: 0.35, fontSize: 17, bold: true });
  T(s, body, { x: x + 0.22, y: 2.5, w: 1.65, h: 1.0, fontSize: 13, color: C.accent3 });
});
card(s, 0.6, 3.85, 8.8, 0.95, C.accent6, "Decision line card");
T(s, "0.176", { x: 0.9, y: 3.97, w: 1.4, h: 0.7, fontSize: 30, bold: true, color: C.accent2, valign: "middle" });
T(s, [
  { text: "Decision line. ", options: { bold: true } },
  { text: "Above this score the AI says “melanoma”. I set it on the tuning images so that 95% of melanomas are caught — fewer misses, more false alarms." },
], { x: 2.35, y: 3.97, w: 6.8, h: 0.7, fontSize: 14, valign: "middle" });
s.addNotes(
  "How did I build it? I used HAM10000, a public dataset of about ten thousand dermoscopy images. " +
  "Dermoscopy means the photo is taken through a skin magnifier. Every image has a diagnosis. " +
  "I split them: about seven thousand to learn from, fifteen hundred to tune, and fifteen hundred kept aside for the final test. " +
  "The model is ResNet-50, a well-known image network. It was already trained on everyday photos, and I retrained it for melanoma. " +
  "Training took about nine minutes on my laptop GPU. " +
  "The model gives a score from zero to one. I chose the decision line, 0.176, so that it catches 95 percent of melanomas. " +
  "The price is more false alarms."
);

// ---------- 4. Results ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("What it achieved on the test images", { placeholder: "title" });
card(s, 0.6, 1.35, 4.2, 2.3, C.text2, "Sensitivity stat");
T(s, "92%", { x: 0.95, y: 1.55, w: 3.5, h: 1.0, fontSize: 60, bold: true, color: C.background1 });
T(s, "of melanomas found", { x: 0.95, y: 2.55, w: 3.5, h: 0.4, fontSize: 18, bold: true, color: C.background1 });
T(s, "157 of 170 test images", { x: 0.95, y: 2.98, w: 3.5, h: 0.35, fontSize: 13, color: C.accent5 });
card(s, 5.2, 1.35, 4.2, 2.3, C.background2, "Specificity stat");
T(s, "66%", { x: 5.55, y: 1.55, w: 3.5, h: 1.0, fontSize: 60, bold: true, color: C.text2 });
T(s, "of non-melanomas cleared", { x: 5.55, y: 2.55, w: 3.5, h: 0.4, fontSize: 18, bold: true });
T(s, "911 of 1,378 — the rest were false alarms", { x: 5.55, y: 2.98, w: 3.5, h: 0.35, fontSize: 13, color: C.accent3 });
T(s, [
  { text: "Trade-off: ", options: { bold: true, color: C.accent1 } },
  { text: "a low decision line means few missed melanomas, but about 1 in 3 non-melanomas gets a false alarm.", options: { breakLine: true } },
  { text: "Photo type matters: ", options: { bold: true, color: C.accent1 } },
  { text: "on ordinary, non-magnified photos it finds far fewer melanomas." },
], { x: 0.6, y: 3.95, w: 8.8, h: 1.0, fontSize: 15, paraSpaceAfter: 8 });
s.addNotes(
  "These are the results on the fifteen hundred test images the model never saw. " +
  "It found 92 percent of melanomas: 157 out of 170. " +
  "It correctly cleared 66 percent of the non-melanomas. The rest were false alarms. " +
  "That is the trade-off of a low decision line: few missed cancers, but about one in three non-melanoma cases gets flagged. " +
  "One more thing I learned: the model was trained on magnified images, and on ordinary photos of the same spots it finds far fewer melanomas."
);

// ---------- 5. Limits and live demo ----------
pres.addSection({ title: "Demo" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo" });
s.addText("Limits, then the live demo", { placeholder: "title" });
const limits = [
  ["Trained only on magnified images", "Ordinary phone photos are less reliable"],
  ["False alarms are common", "The low decision line is a deliberate choice"],
  ["The heatmap is not proof", "It shows where the AI looked, not why it is right"],
  ["Not a medical tool", "A learning project, not tested in a clinic"],
];
limits.forEach(([head, body], i) => {
  const y = 1.35 + i * 0.85;
  badge(s, 0.6, y + 0.05, "!", C.accent6, C.accent2, `Limit ${i + 1} badge`);
  T(s, head, { x: 1.2, y, w: 3.9, h: 0.32, fontSize: 16, bold: true });
  T(s, body, { x: 1.2, y: y + 0.33, w: 3.9, h: 0.32, fontSize: 13, color: C.accent3 });
});
card(s, 5.5, 1.35, 3.9, 3.35, C.text2, "Demo card");
T(s, "LIVE DEMO", { x: 5.85, y: 1.6, w: 3.2, h: 0.3, fontSize: 12, bold: true, color: C.accent5, charSpacing: 1 });
T(s, "Three test images", { x: 5.85, y: 1.95, w: 3.2, h: 0.45, fontSize: 21, bold: true, color: C.background1 });
[["A melanoma", "caught"], ["A non-melanoma", "cleared"], ["A common mole", "false alarm"]].forEach(([a, b], i) => {
  const y = 2.6 + i * 0.62;
  badge(s, 5.85, y, String(i + 1), C.background1, C.text2, `Demo step ${i + 1}`);
  T(s, [{ text: a, options: { bold: true } }, { text: " → " + b }], { x: 6.4, y, w: 2.85, h: 0.42, fontSize: 15, color: C.background1, valign: "middle" });
});
s.addNotes(
  "Before the demo, four honest limits. " +
  "It was trained only on magnified images, so ordinary phone photos are less reliable. " +
  "False alarms are common, and that is a choice I made. " +
  "The heatmap shows where the AI looked, not proof that it is right. " +
  "And it is a learning project, not a medical tool. " +
  "Now let me show you three test images: a melanoma it catches, a non-melanoma it clears, and a common mole where it gives a false alarm."
);

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
