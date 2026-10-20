# Speech — Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection (about 3 minutes + 2-minute live demo)

Speak slowly. The same text is in each slide's Speaker notes (PowerPoint: View → Notes, or Presenter View).

## Slide 1 — Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection  (0:00–0:25)

Hello everyone. My research is called Cross-Modality Robustness of Explainable Deep Learning for Melanoma Detection. In simple words: I built an AI that looks at a close-up image of a skin spot and decides melanoma or not melanoma, and I checked whether it still works, and whether its explanations still make sense, when the image type changes. I'll take about three minutes to explain the idea, the data and the results, and then I'll show you the live demo.

## Slide 2 — Why this project  (0:25–0:55)

Why this project? Melanoma is a dangerous type of skin cancer, and finding it early matters. But many skin spots look very similar. So my goal was an AI that answers one simple question: is this melanoma or not? It also shows where in the image it looked, and it runs in a small web app that I'll show you.

## Slide 3 — How I built it  (0:55–1:50)

How did I build it? I used HAM10000, a public dataset of about ten thousand dermoscopy images. Dermoscopy means the photo is taken through a skin magnifier. Every image has a diagnosis. I split them: about seven thousand to learn from, fifteen hundred to tune, and fifteen hundred kept aside for the final test. The model is ResNet-50, a well-known image network. It was already trained on everyday photos, and I retrained it for melanoma. Training took about nine minutes on my laptop GPU. The model gives a score from zero to one. I chose the decision line, 0.176, so that it catches 95 percent of melanomas. The price is more false alarms.

## Slide 4 — What it achieved on the test images  (1:50–2:30)

These are the results on the fifteen hundred test images the model never saw. It found 92 percent of melanomas: 157 out of 170. It correctly cleared 66 percent of the non-melanomas. The rest were false alarms. That is the trade-off of a low decision line: few missed cancers, but about one in three non-melanoma cases gets flagged. One more thing I learned: the model was trained on magnified images, and on ordinary photos of the same spots it finds far fewer melanomas.

## Slide 5 — Limits, then the live demo  (2:30–3:00)

Before the demo, four honest limits. It was trained only on magnified images, so ordinary phone photos are less reliable. False alarms are common, and that is a choice I made. The heatmap shows where the AI looked, not proof that it is right. And it is a learning project, not a medical tool. Now let me show you three test images: a melanoma it catches, a non-melanoma it clears, and a common mole where it gives a false alarm.

## Live demo (about 2 minutes)

Switch to the browser (http://127.0.0.1:8765). Follow `demo/DEMO_CHEATSHEET_EN.md`:
1. Example **02** → "Looks like melanoma", score 0.99 — "A real melanoma: caught."
2. Example **05** → "Does not look like melanoma", score about 0 — "A non-melanoma: cleared."
3. Example **06** → "Looks like melanoma", score 0.94, true diagnosis common mole — "This is a false alarm. The AI makes mistakes, and false alarms are the most common one."

Close: "That's my project. Thank you — I'm happy to take questions."

_Slide speech: 415 words ≈ 3.0 minutes at a calm pace._
