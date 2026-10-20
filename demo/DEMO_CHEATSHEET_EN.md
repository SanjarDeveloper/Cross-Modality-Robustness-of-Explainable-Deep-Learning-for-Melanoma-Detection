# Demo cheat sheet (5 minutes)

Start: double-click `start_demo.bat`, wait for **"Model loaded and verified."** in the server window,
then open http://127.0.0.1:8765. Do this **before** the meeting starts. Keep the server window open.

---

## 1. Four ideas you must be able to explain

**Score**
A number from 0 to 1 that the AI gives each image. Higher means "more melanoma-like".
It is *not* a percentage chance of cancer. Say: *"It's how melanoma-like the image looks to the model, not a medical probability."*

**Decision line (threshold = 0.176)**
If the score is above the line, the AI says "looks like melanoma".
The line is set **low on purpose**: I picked it on separate validation images so the AI catches 95% of melanomas.
The price is more false alarms. Say: *"Missing a melanoma is worse than a false alarm, so the line is low."*

**Heatmap (Grad-CAM)**
Shows which part of the image pushed the score up. Red = where the AI looked most.
Colours are relative inside each image, so there is always a "hottest" spot, even when the score is tiny.
Say: *"It shows where the model looked, not proof that it is right."*

**"Not melanoma" ≠ "safe"**
The other class includes moles, but also other skin cancers. The AI only answers one question: melanoma or not.

---

## 2. Five-minute script

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the title | "This is a skin lesion classifier: it looks at a close-up skin image and says melanoma or non-melanoma. Melanoma is a dangerous skin cancer. It also shows where it looked." |
| 0:40 | Click example **02**, press **Check image** | "This is a real melanoma the AI never saw in training. Score 0.99, far above the line — it says melanoma, and the heatmap is on the lesion." |
| 1:30 | Click example **05**, press **Check image** | "This one is not melanoma. Score almost zero — correct." |
| 2:00 | Click example **06**, press **Check image** | "The AI also makes mistakes. This is a normal mole, but it scored 0.94 — a false alarm. Because the line is set low, false alarms are the most common mistake." |
| 2:50 | Point at the two result images | "How it works, in five steps: the image is resized to 224 by 224, a ResNet-50 model gives a score, the score is compared with the decision line, and Grad-CAM draws the heatmap." |
| 3:40 | Point at the example images | "It learned from HAM10000, about ten thousand dermoscopy images. About seven thousand for learning; these examples are from the separate test set it never saw." |
| 4:20 | — | "It runs only on this laptop, uploads are not saved, and it's a research demo, not a medical tool. Questions?" |

Tip: if the GPU is slow the first time, talk over it. Each check takes a second or two.

---

## 3. Likely questions and short answers

**How accurate is it?**
On 1,548 test images it found 92% of melanomas (157 of 170). It correctly cleared 66% of non-melanomas, so about one in three non-melanomas gets a false alarm. That is the trade-off of the low decision line.

**Why ResNet-50?**
It is a well-known, reliable image model. I started from a version already trained on general photos (ImageNet) and retrained it for melanoma. I also compared it with EfficientNet-B3 and DenseNet-121; none was clearly better overall.

**Does it work on phone photos?**
Less well. It was trained on dermoscopy images, taken through a skin magnifier. On ordinary (non-magnified) photos of the same lesions it found far fewer melanomas. That gap was the main topic of my research.

**How long did training take, and on what?**
About 9–10 minutes for 8 passes over the training images, on a laptop GPU (RTX 5060).

**Where does the data come from?**
HAM10000, a public dataset of 10,015 dermoscopy images with diagnoses. Licence CC BY-NC 4.0 (non-commercial).

**Is the heatmap reliable?**
It shows where the model looked, not *why* it is right. It can highlight the wrong place, as in the false-alarm example. In my tests, a simple "look at the centre of the image" rule found the lesion better than the heatmaps, because lesions are usually photographed in the middle.

**Is my uploaded image stored?**
No. It is processed in memory on this laptop and never written to disk or sent anywhere.

**What is the tech stack?**
PyTorch and timm for the model, OpenCV for resizing and heatmap colours, a small Python web server, and plain HTML/CSS/JavaScript for the page.

**Could a doctor use this?**
No. It was not tested in a clinic, it was not approved as a medical device, and the score is not a calibrated probability.

**What would you do next?**
Test on more real phone photos, and train with phone-style images so the model handles them better.

**Did you write all of this yourself?** *(only if asked directly — your choice of wording)*
Answer honestly. For example: *"I planned and directed the project step by step and used AI coding assistants to write much of the code; I checked the results against saved outputs."*

---

## 4. If something goes wrong

| Problem | Fix |
|---|---|
| "Cannot reach the server" error | Server window was closed: run `start_demo.bat` again |
| "Port already in use" | Close the old server window, or run with `--port 8766` (see README) and open that port |
| Uploaded image rejected | Use JPEG/PNG under 10 MB, no transparency, already upright |
| Page looks old after changes | Press Ctrl+F5 |
