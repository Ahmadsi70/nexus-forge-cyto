# استراتژی انتشار — Nexus-Forge Cyto
## همهچیز رایگان، با حفظ تمامی حقوق پروژه

---

## ۱. مدل حقوقی: "Source Available — All Rights Reserved"

**اصل:** همه می‌توانند رایگان استفاده کنند، ولی مالکیت پروژه کاملاً برای ClinicalGuard محفوظ است.

```
┌──────────────────────────────────────────────────────────┐
│  مالک: ClinicalGuard (شما)                                │
│  لایسنس: Custom — "Free Usage, All Rights Reserved"       │
│                                                          │
│  ✅ هرکسی می‌تواند رایگان استفاده کند (شخصی، علمی، بالینی) │
│  ✅ سورس در GitHub قابل مشاهده است (Source Available)     │
│  ✅ شرکتها می‌توانند رایگان استفاده کنند (بدون محدودیت)    │
│  ✅ هیچ‌کس مجوز فروش، sublicense، یا ادعای مالکیت ندارد    │
│  ❌ هیچ‌کس نمی‌تواند پروژه را fork کرده و ادعای مالکیت کند  │
│  ❌ هیچ‌کس نمی‌تواند پروژه را تحت نام خود منتشر کند        │
│  ❌ هیچ‌کس نمی‌تواند از نام Nexus-Forge / ClinicalGuard    │
│     برای محصول خود استفاده کند                            │
└──────────────────────────────────────────────────────────┘
```

---

## ۲. تفاوت با مدلهای رایج

| مدل | سورس | استفاده تجاری | فروش توسط دیگران | مالکیت |
|-----|------|--------------|-----------------|--------|
| **MIT** | ✅ باز | ✅ آزاد | ✅ آزاد | ❌ واگذار میشود |
| **GPL** | ✅ باز | ✅ آزاد | ❌ باید متنباز کنند | ❌ محدود |
| **BSL 1.1** | ✅ باز | ❌ نیاز به مجوز | ❌ ممنوع | ✅ شما |
| **Proprietary** | ❌ بسته | ❌ نیاز به مجوز | ❌ ممنوع | ✅ شما |
| **✅ مدل ما (Custom)** | ✅ باز | ✅ رایگان | ❌ ممنوع | ✅ شما |

**نتیجه:** مدل ما唯一 مدلی است که هم سورس باز است، هم استفاده تجاری رایگان است، هم مالکیت را حفظ میکند.

---

## ۳. متن لایسنس اختصاصی (Custom License)

فایل `LICENSE` در روت پروژه:

```text
Nexus-Forge Cyto — License
Copyright (c) 2024 ClinicalGuard. All Rights Reserved.

TERMS AND CONDITIONS

1. GRANT OF FREE USAGE
   ClinicalGuard grants you a non-exclusive, non-transferable,
   worldwide, royalty-free license to:
   
   a) Use the software for any lawful purpose, including but not
      limited to: academic research, clinical diagnostics, commercial
      products, SaaS platforms, and personal projects.
   
   b) Copy, modify, and create derivative works for your own use.
   
   c) Distribute the software in its original or modified form,
      provided that:
      - You do NOT charge any fee for the software itself
      - You do NOT sublicense or resell the software
      - You retain all copyright notices
      - You clearly state that modifications are your own

2. RESERVATION OF RIGHTS
   All rights not expressly granted above are reserved by ClinicalGuard.
   This includes but is not limited to:
   
   a) The right to sell commercial licenses (though we choose not to)
   b) The right to change the license in future versions
   c) The right to use the software in any way ClinicalGuard sees fit
   d) Ownership of all copyright, trademark, and patent rights

3. TRADEMARK RESTRICTIONS
   You may NOT use the names "Nexus-Forge", "ClinicalGuard", or any
   related logos in a way that suggests endorsement or association
   without prior written permission.

4. NO WARRANTY
   The software is provided "AS IS", without warranty of any kind.
   ClinicalGuard is not liable for any damages arising from its use.

5. ACCEPTANCE
   By using the software, you accept these terms. If you do not agree,
   do not use the software.

6. GOVERNING LAW
   This license is governed by the laws of Iran, with dispute
   resolution in Tehran.
```

---

## ۴. فایلهای حقوقی در GitHub

```
cancer_project/
├── LICENSE                    # Custom License — All Rights Reserved
├── NOTICE.md                  # اطلاعیهها و اعتبارنامهها
├── TRADEMARK.md               # خطمشی استفاده از نام برند
└── SECURITY.md                # گزارش آسیبپذیریها
```

**توجه: هیچ CLA, COMMERCIAL.md, یا BSL نیاز نیست.** چون همهچیز رایگان است، فقط یک فایل LICENSE کافی است.

---

## ۵. متن `NOTICE.md`

```markdown
# Notice

## Copyright
Copyright (c) 2024 ClinicalGuard. All Rights Reserved.

## Third-Party Components
This project includes code from the following open-source projects,
each under its own license:

- **HoVerNet** — Original architecture by Graham et al. (MIT License)
- **ONNX Runtime** — Microsoft (MIT License)
- **PyTorch** — Linux Foundation (BSD License)
- **tract** — Tract project (Apache 2.0 / MIT)
- **Axum** — Tokio project (MIT License)

## Dataset Acknowledgments
- **PanNuke** — CC BY-NC-SA 4.0 (Gamper et al.)
- **TopoDiff** — MIT dataset (synthetic, no restrictions)
- **Cantilever** — Synthetic displacement dataset
- **Pereira** — Public dataset

## Contact
ClinicalGuard — [your-telegram/email]
```

---

## ۶. متن `TRADEMARK.md`

```markdown
# Trademark Guidelines

"Nexus-Forge" and "ClinicalGuard" are trademarks of ClinicalGuard.

## Permitted Use
- Referring to the project in publications, presentations, and discussions
- Using the name in your own fork with clear attribution
- Mentioning "Powered by Nexus-Forge" in your product

## Prohibited Use
- Using "Nexus-Forge" or "ClinicalGuard" as your product or company name
- Registering these names as domain names
- Implying endorsement or partnership with ClinicalGuard
- Removing trademark notices from the code
```

---

## ۷. Badgeهای README

```markdown
![License: Custom](https://img.shields.io/badge/License-Custom%20(All%20Rights%20Reserved)-blue)
![Free](https://img.shields.io/badge/Free-For%20All%20Uses-brightgreen)
![Source Available](https://img.shields.io/badge/Source-Available-lightgrey)
```

---

## ۸. پیام در README (بخش اول)

```markdown
# 🧬 Nexus-Forge Cyto

**Explainable Geometry for Digital Pathology**

**100% Free — No Commercial License Required**

Nexus-Forge is completely free to use for any purpose:
- ✅ Academic research
- ✅ Clinical diagnostics
- ✅ Commercial products
- ✅ SaaS platforms
- ✅ Personal projects

All source code is available on GitHub. The project is owned by
**ClinicalGuard** and distributed under a custom license that
grants free usage while reserving all ownership rights.

You may use, modify, and distribute the software freely, but you
may NOT sell, sublicense, or claim ownership of it.
```

---

## ۹. استراتژی Contributorها

از آنجا که لایسنس **غیر استاندارد** است، Contributors باید صریحاً با آن موافق باشند:

### در Pull Request Template:

```markdown
By submitting this Pull Request, I confirm that:
1. My contribution is my own work (or I have permission)
2. I agree to my contribution being distributed under the project's
   existing license (Custom — All Rights Reserved)
3. I understand that ClinicalGuard retains all rights to the project
```

### رویه:

```
شخص Fork میکند ← Pull Request میزند ←
Template بالا را تأیید میکند ← شما Merge میکنید
```

اگر کسی با لایسنس موافق نباشد، نمی‌تواند contribute کند.

---

## ۱۰. شفافیت درباره مدلهای PanNuke

PanNuke تحت **CC BY-NC-SA 4.0** است — یعنی استفاده غیرتجاری. این محدودیت از طرف PanNuke است، نه از طرف شما.

در `NOTICE.md` و README قید کنید:

```markdown
⚠️ **PanNuke-based models:** Some model weights in this repository
are derived from the PanNuke dataset (CC BY-NC-SA 4.0). While the
Nexus-Forge license permits commercial use, the PanNuke dataset
itself restricts commercial use. Users are responsible for complying
with PanNuke's license terms. Models trained on synthetic data
(TopoNet) have no such restriction.
```

---

## ۱۱. مزایای این مدل

| مزیت | توضیح |
|------|-------|
| **Adoption حداکثر** | چون همهچیز رایگان است، هیچ مانعی برای استفاده نیست |
| **شفافیت کامل** | سورس باز است، پزشکان اعتماد دارند |
| **حفظ مالکیت** | هیچ‌کس نمی‌تواند ادعای مالکیت کند یا پروژه را بفروشد |
| **بدون پیچیدگی حقوقی** | یک فایل LICENSE کافی است — نیازی به CLA, BSL, Commercial نیست |
| **قابلیت تغییر لایسنس آینده** | چون مالکیت محفوظ است، در آینده می‌توانید لایسنس را تغییر دهید (مثلاً اگر شرکت بزرگ شد) |
| **اعتبار برند** | ClinicalGuard به عنوان مالک و خالق پروژه شناخته میشود |

---

## ۱۲. ریسکها و راهکارها

| ریسک | راهکار |
|------|--------|
| شخصی کد را fork کرده و ادعای مالکیت کند | لایسنس صراحتاً می‌گوید All Rights Reserved. اگر ادعا کند، پیگرد قانونی. |
| شخصی کد را بفروشد | لایسنس فروش را ممنوع کرده. در عمل کسی برای کد رایگان پول نمی‌دهد. |
| شرکتی از نام Nexus-Forge برای محصول خود استفاده کند | Trademark این را ممنوع کرده. |
| Contributor از سورس استفاده نامناسبی کند | با PR Template تأیید میکند. بدون آن نمی‌تواند contribute کند. |
| لایسنس غیراستاندارد باعث سردرگمی شود | در README واضح توضیح داده شده: "Free to use, but we own it." |

---

## ۱۳. خلاصه — یک پاراگراف

> این پروژه **کاملاً رایگان** است برای همه، بدون استثنا. سورس در GitHub قابل مشاهده است. هرکسی می‌تواند استفاده کند، کپی کند، تغییر دهد، و توزیع کند — **به شرطی که نفروشد و ادعای مالکیت نکند.** مالکیت پروژه برای ClinicalGuard محفوظ است. این ساده‌ترین و شفاف‌ترین مدل حقوقی است: همهچیز رایگان، همهچیز شفاف، مالکیت با شما.