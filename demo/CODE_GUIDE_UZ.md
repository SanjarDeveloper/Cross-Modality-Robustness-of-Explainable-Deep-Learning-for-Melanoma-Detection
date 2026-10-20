# Kodni o‘rganish: sodda yo‘l

## 1. Ishga tushirish
`start_demo.bat` ustiga ikki marta bosing. Oyna ochiq tursin.
Brauzerda http://127.0.0.1:8765 ni oching. Model server ochilganda fonda yuklanadi;
yuqoridagi belgi yashil bo‘lsa ("Model ready"), tayyor.
To‘xtatish: server oynasida Ctrl+C. Port band bo‘lsa README’dagi boshqa portni ishlating.

## 2. Qaysi faylni o‘zgartiraman?
**Matn yoki tugma:** `frontend/index.html`. HTML sahifaning tuzilishi.
**Rang, shrift, joylashuv:** `frontend/styles.css`. Boshidagi `--accent` asosiy rang.
**Tugma nima qiladi:** `frontend/app.js`. JavaScript foydalanuvchi harakatini boshqaradi.
Frontend o‘zgargach brauzerni Ctrl+F5 bilan yangilang; Python bilish shart emas.

## 3. Rasmning yo‘li
1. Namuna rasm bosiladi (`/samples/01.jpg` …) yoki fayl tanlanadi; "Run model" bosilganda
   `app.js` uni `POST /api/predict` ga yuboradi.
2. `app.py` fayl hajmini tekshiradi va `model_service.py` ni chaqiradi.
3. Rasm RGB rangga, 256 pikselga, so‘ng 224 pikselga o‘tadi.
4. Research kodidagi normalizatsiya va model ishlaydi.
5. Bal va Grad-CAM xaritasi JSON javobida qaytadi.
6. JavaScript natijani sahifaga qo‘yadi. Yuklangan fayl diskka yozilmaydi.

## 4. Natijalar sahifasi
`GET /api/results` thesis’dagi `MASTER_RESULTS_v2.csv` ni o‘qiydi.
"Research results" tabida 4 ta asosiy natija kartasi, 4 ta grafik va to‘liq jadval bor.
RQ tugmalari va qidiruv qatorlarni saralaydi. Bu qayta hisoblangan natija emas.
N — tanlama soni; klaster — birga qayta tanlangan guruh. Har qator manbasi ko‘rsatiladi.

## 5. Nimalarni o‘zgartirmaslik kerak?
Model checkpoint’i, threshold, preprocessing va research fayllarini UI dizayni
uchun o‘zgartirmang. Ular ilmiy natijalarni aniqlaydi. Backend’ni o‘zgartirsangiz,
serverni qayta ishga tushiring. Koddagi izohlar ingliz tilida yozilgan.

## 6. Natijani tushunish
Bal saratonning klinik ehtimoli emas. “Melanoma emas” xavfsiz degani emas.
Yorqin Grad-CAM joylari modelning musbat xaritasini ko‘rsatadi, klinik isbot emas.
Nol xarita chiqishi mumkin; dastur uni sun’iy rang bilan almashtirmaydi.
Grad-CAM ranglari har bir rasm ichida o‘lchanadi: past bal bo‘lsa ham eng "issiq" joy chiqadi,
bu kuchli dalil degani emas.
