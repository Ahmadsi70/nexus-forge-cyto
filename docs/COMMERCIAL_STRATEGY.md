# استراتژی تجاری و حق مالکیت — Nexus-Forge Cyto
## انتشار در GitHub با حفظ حقوق تجاری

---

## ۱. چارچوب حقوقی انتشار

### ۱.۱. مالکیت پروژه

```
پروژه:          Nexus-Forge Cyto
مالک:           ClinicalGuard (شما)
حوزه:           ایران / بینالملل
نوع دارایی:     سورس کد + الگوریتمها + مدلهای آموزشدیده + برند
```

تمامی حقوق مادی و معنوی پروژه نزد شما محفوظ است. انتشار در GitHub **به معنای واگذاری حقوق نیست**.

### ۱.۲. تفکیک مالکیت بر اساس نوع دارایی

| نوع دارایی | مالک | لایسنس انتشار | توضیح |
|-----------|------|-------------|-------|
| **سورس کد ابزارها** (Python, Groovy, YAML, Docker) | شما | MIT | کاربر می‌تواند کپی کند، تغییر دهد، حتی بفروشد — اینها مزیت رقابتی نیستند |
| **سورس کد هسته هندسه** (Rust geometry engine) | شما | **BSL 1.1** | سورس باز است ولی فروش ممنوع. شرکتها برای استفاده تجاری باید License بخرند |
| **مدلهای آموزشدیده** (ONNX weights) | شما | **Proprietary** | فقط باینری منتشر میشود. سورس وزنها محرمانه است |
| **نام + برند + لوگو** | شما | ** trademark** | هیچ‌کس نمی‌تواند از نام Nexus-Forge یا ClinicalGuard برای محصول خود استفاده کند |
| **مستندات** | شما | MIT | مستندات را هرکسی می‌تواند کپی کند |
| **Pull Requests از Contributors** | شما + contributor | تحت CLA | با امضای CLA، شما حق دارید contributions را در Commercial License مجدداً مجوز دهید |

---

## ۲. ساختار فایلهای حقوقی در GitHub

در روت پروژه، این فایل‌ها قرار میگیرند:

```
cancer_project/
├── LICENSE                      # MIT — برای ابزارها
├── LICENSE.bsl                  # BSL 1.1 — برای هسته هندسه
├── LICENSE.models               # Proprietary — برای مدلها
├── COMMERCIAL.md                # راهنمای خرید مجوز تجاری
├── NOTICE.md                    # شفافیت درباره مدلهای شخص ثالث
├── CLA.md                       # Contributor License Agreement
├── TRADEMARK.md                 # خطمشی استفاده از نام/برند
└── SECURITY.md                  # گزارش آسیبپذیریها
```

---

## ۳. متن کامل هر فایل

### ۳.۱. `LICENSE` (MIT — برای ابزارها و اسکریپتها)

```text
MIT License

Copyright (c) 2024 ClinicalGuard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ۳.۲. `LICENSE.bsl` (BSL 1.1 — برای هسته هندسه)

```text
Business Source License 1.1

License text copyright (c) 2017 MariaDB Corporation Ab, All Rights Reserved.
"Business Source License" is a trademark of MariaDB Corporation Ab.

-----------------------------------------------------------------------------
Parameters

Licensor:             ClinicalGuard
Licensed Work:        Nexus-Forge Cyto Geometry Engine
                      The Licensed Work is (c) 2024 ClinicalGuard
Additional Use Grant: Any use cases not expressly prohibited by this license,
                      including but not limited to: non-commercial academic
                      research, clinical diagnostics, personal/educational use,
                      and evaluation purposes. A for-profit entity using the
                      Licensed Work in a revenue-generating product, service,
                      or SaaS platform must obtain a commercial license from
                      the Licensor.

Change Date:          Four years from the date the Licensed Work is first
                      distributed by the Licensor.
Change License:       Apache License, Version 2.0

-----------------------------------------------------------------------------

Terms

The Licensor hereby grants you the right to copy, modify, create derivative
works, publicly display, and distribute the Licensed Work under the following
conditions:

1. You may not use the Licensed Work for any purpose that generates revenue,
   directly or indirectly, without a separate commercial license from the
   Licensor.
2. You must include a copy of this license with every copy of the Licensed
   Work you distribute.
3. You may not remove or alter any license notices contained in the Licensed
   Work.
4. The Licensed Work is provided "AS IS", without warranty of any kind.

After the Change Date, the Licensed Work becomes available under the Change
License specified above.
```

> **نکته مهم:** فایل `LICENSE.bsl` را در پوشه `nexus-forge-cyto-geometry/` قرار میدهید (نه در روت پروژه)، چون فقط آن crate تحت BSL است.

### ۳.۳. `LICENSE.models` (Proprietary — برای مدلها)

```text
Nexus-Forge Cyto — Model Weights License
Copyright (c) 2024 ClinicalGuard. All Rights Reserved.

TERMS AND CONDITIONS

1. GRANT OF LICENSE
   ClinicalGuard grants you a non-exclusive, non-transferable,
   royalty-free license to use the model weights ("Models") for:
   - Non-commercial academic research
   - Clinical diagnostics and patient care
   - Personal and educational purposes

2. PROHIBITED USES
   You may NOT:
   a) Use the Models in any commercial product or service
   b) Offer the Models as part of a SaaS platform
   c) Sell, sublicense, or distribute the Models for profit
   d) Use the Models in a manner that competes with ClinicalGuard

3. THIRD-PARTY DATA
   Some Models may incorporate weights derived from datasets under
   separate licenses (e.g., PanNuke CC BY-NC-SA 4.0). You must comply
   with all applicable third-party license terms.

4. COMMERCIAL LICENSING
   For commercial use, contact ClinicalGuard at [email/telegram] to
   obtain a separate commercial license.

5. DISCLAIMER
   The Models are provided "AS IS" without warranty. ClinicalGuard is
   not liable for any damages arising from their use.
```

### ۳.۴. `COMMERCIAL.md` — مهمترین فایل از نظر تجاری

```markdown
# Commercial Licensing — Nexus-Forge Cyto

## Why a Commercial License?

Nexus-Forge Cyto is free for non-commercial use (academic research,
clinical diagnostics, personal projects). If your organization generates
revenue from using or integrating Nexus-Forge, you need a commercial license.

## Who Needs a License?

| Scenario | License Required? |
|----------|------------------|
| University research lab | ❌ Free (BSL/MIT) |
| Hospital diagnostics (non-profit) | ❌ Free (BSL/MIT) |
| Hospital diagnostics (for-profit) | ✅ Commercial License |
| Startup embedding in app | ✅ Commercial License |
| SaaS platform offering analysis | ✅ Commercial License |
| Enterprise internal tool | ⚠️ Depends on revenue impact |
| Consulting using Nexus-Forge | ✅ Commercial License |

## What You Get

- Full source code access (including Rust geometry engine)
- Priority email/Telegram support
- Commercial indemnification
- Custom integration assistance (optional)
- License covers all past and future versions

## Pricing

| Tier | Organization Size | Annual Fee | Includes |
|------|------------------|-----------|----------|
| **Startup** | ≤ 10 employees | **$500** | Full access, email support |
| **Small Business** | ≤ 50 employees | **$2,000** | Full access, priority support |
| **Enterprise** | Unlimited | **Custom** | Full access, dedicated support, SLA |
| **SaaS / Cloud** | Per-revenue | **5% rev share** | Full access, integration help |
| **OEM / Embedding** | Per-device | **Negotiable** | Royalty-based or flat fee |
| **Reseller / White-label** | Per-client | **Negotiable** | Resell under your brand |

> All prices are in USD. Discounts available for startups in low-income countries.

## How to Purchase

1. Contact: [your-telegram] or [your-email]
2. Tell us your use case and organization size
3. We issue an invoice (PayPal / Wire / Crypto)
4. You receive the commercial license certificate + additional materials

## Frequently Asked Questions

**Q: Can I try before buying?**
A: Yes. The BSL license allows evaluation and prototyping. You only need
   a commercial license when you go to production or generate revenue.

**Q: What if I accidentally violate the license?**
A: Contact us. We prefer compliance over litigation. We'll work out a
   retroactive license if needed.

**Q: Can I use the MIT-licensed tools commercially for free?**
A: Yes! The MIT layer (QuPath extension, CLI, Streamlit, etc.) is fully
   free for any use, including commercial. Only the geometry engine (BSL)
   and models (Proprietary) require a license.

**Q: Do I need a license if I'm a student?**
A: No. All layers are free for non-commercial academic use.
```

---

## ۴. استراتژی Contributor Management

### ۴.۱. `CLA.md` (Contributor License Agreement)

این فایل را در روت پروژه قرار دهید. هر Contributor قبل از اولین Pull Request باید آن را امضا کند:

```markdown
# Contributor License Agreement (CLA)

Thank you for contributing to Nexus-Forge Cyto. By submitting a Pull Request,
you agree to the following terms:

## Grant of Rights

1. **Copyright License:** You grant ClinicalGuard a perpetual, worldwide,
   non-exclusive, royalty-free license to use, copy, modify, distribute,
   and sublicense your contribution under any license, including commercial
   licenses.

2. **Patent License:** You grant ClinicalGuard a license to any patents
   that your contribution infringes, for the purpose of distributing the
   project.

3. **Moral Rights:** To the extent permitted by law, you waive any moral
   rights in your contribution.

## Representations

1. You own the contribution or have permission from its owner.
2. The contribution does not infringe any third-party rights.
3. You are legally entitled to grant this license.

## No Obligation

ClinicalGuard is not obligated to use your contribution.

## How to Sign

Add a comment to your Pull Request saying:
> "I have read the CLA and hereby sign it."
```

### ۴.۲. رویه عملی برای Contributors

```
شخص Fork میکند ← Pull Request میزند ← 
ربات CLA را یادآوری میکند ← شخص کامنت میگذارد ← 
شما Merge میکنید

اگر شخص حاضر به امضای CLA نباشد → Pull Request را ببندید
```

---

## ۵. استراتژی قیمتگذاری دقیق

### ۵.۱. منطق قیمت

| فاکتور | تأثیر بر قیمت |
|--------|--------------|
| **هزینه توسعه** (تخمینی ~$50,000+) | قیمت پایه را تعیین میکند |
| **ارزش برای شرکت** (جایگزین ابزار $10,000+/سال) | قیمت سقف را تعیین میکند |
| **قابلیت جایگزینی** (رقیب میتواند ۳ ماهه بازنویسی کند) | قیمت را محدود میکند |
| **قدرت برند** (ابتدای کار = صفر) | قیمت را پایین نگه میدارد |
| **اندازه بازار** (هزاران کلینیک + شرکتهای مهندسی) | تخفیف برای adoption |

### ۵.۲. قیمتهای پیشنهادی (مرحله اول)

| سطح | قیمت | مخاطب هدف |
|-----|------|-----------|
| **رایگان** | $0 | دانشگاهها، محققین، استفاده شخصی |
| **Startup** | **$500/سال** | شرکتهای <۱۰ نفر — قیمت سمبلیک برای شروع |
| **Business** | **$2,000/سال** | شرکتهای ۱۰-۵۰ نفر — قیمت اصلی |
| **Enterprise** | **$5,000-$20,000/سال** | سازمانهای بزرگ — negotiate |
| **SaaS** | **۵٪ از درآمد** | پلتفرمهای ابری — revenue share |
| **OEM** | **$0.10-$1.00 per analysis** | شرکتهای محصولی — royalty |

### ۵.۳. تخفیفهای استراتژیک

| وضعیت | تخفیف | دلیل |
|-------|-------|------|
| شرکت ایرانی | **۵۰٪** | Supporting local industry |
| استارتاپ <۲ سال | **۳۰٪** | Ecosystem investment |
| Contributor فعال | **رایگان ۱ سال** | قدردانی |
| Proof-of-Concept (محدود ۳ ماه) | **رایگان** | فروش آینده |

---

## ۶. GitHub Repository Setup

### ۶.۱. تنظیمات پیشنهادی

| تنظیم | مقدار | دلیل |
|-------|-------|------|
| Visibility | **Public** | adoption حداکثر |
| Default branch | `main` | استاندارد |
| Branch protection | ✅ Require PR review | کیفیت کد |
| Issues | ✅ Enabled | بازخورد |
| Discussions | ✅ Enabled | جامعه |
| Wiki | ❌ Disabled | docs در خود repo |
| Sponsors | ❌ Disabled | تمرکز روی Commercial License |
| Template for Issues | ✅ Bug + Feature + License | استانداردسازی |

### ۶.۲. README Badgeها

```markdown
![License: Tools](https://img.shields.io/badge/Tools-MIT-green)
![License: Engine](https://img.shields.io/badge/Engine-BSL%201.1-blue)
![License: Models](https://img.shields.io/badge/Models-Proprietary-red)
![Commercial](https://img.shields.io/badge/Commercial%20License-Required-orange)
![CI](https://img.shields.io/github/actions/workflow/status/.../ci.yml?branch=main)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Rust](https://img.shields.io/badge/Rust-2021-orange)
```

---

## ۷. حفاظت از برند (Trademark)

### ۷.۱. `TRADEMARK.md`

```markdown
# Trademark Guidelines

"Nexus-Forge" and "ClinicalGuard" are trademarks of ClinicalGuard.

## What You Can Do
- Use the name to refer to the project in publications, presentations
- Fork the repository and keep the name in your fork
- Use the name in non-commercial settings

## What You Cannot Do
- Use "Nexus-Forge" or "ClinicalGuard" as your product or company name
- Use the name in a way that suggests endorsement
- Remove trademark notices from the code
- Sell a product called "Nexus-Forge" without a license
```

### ۷.۲. ثبت برند (توصیه آینده)

| مرحله | هزینه | زمان |
|-------|-------|------|
| ثبت دامنه `nexus-forge.org` | ~$15/سال | امروز |
| ثبت علامت تجاری ایران | ~$50 | ۳-۶ ماه |
| ثبت علامت تجاری بینالملل (WIPO) | ~$500+ | ۶-۱۲ ماه |

---

## ۸. Risk Mitigation

| ریسک | احتمال | راهکار |
|------|--------|--------|
| شرکتی کد را fork کند و بدون مجوز استفاده کند | متوسط | **BSL** این را صراحتاً ممنوع کرده. حق پیگیری قانونی دارید |
| Contributor سورس را Fork کرده و علیه شما استفاده کند | کم | **CLA** این را پوشش میدهد — شما حق دارید contributions را مجدداً مجوز دهید |
| Elastic-style fork توسط رقیب | کم | Brand power + data + integration = moat واقعی |
| شخصی مدل را reverse-engineer کند | خیلی کم | مدل ۳۷ میلیون پارامتر است — reverse engineering ارزش اقتصادی ندارد |
| شرکت خارج از آمریکا/اروپا BSL را نادیده بگیرد | زیاد (در عمل) | اگر حقوقی قابل پیگیری نیست، روی **ابزارهای MIT** تمرکز کنید که adoption بیاورند. شرکتها در نهایت برای پشتیبانی و اعتبار خرید میکنند |

---

## ۹. برنامه اقدام — امروز

| مرحله | کار | زمان |
|-------|-----|------|
| ۱ | ایجاد فایلهای LICENSE, LICENSE.bsl, LICENSE.models, COMMERCIAL.md, CLA.md, TRADEMARK.md | ۳۰ دقیقه |
| ۲ | اضافه کردن badgeها به README | ۱۰ دقیقه |
| ۳ | Push به GitHub | ۵ دقیقه |
| ۴ | ایجاد اولین Release (v0.1.0) با باینریهای هسته | ۳۰ دقیقه |
| ۵ | تنظیم GitHub Branch Protection | ۱۰ دقیقه |
| ۶ | نوشتن Issue/PR templates | ۱۵ دقیقه |

مجموع: **حدود ۱.۵ ساعت** تا انتشار کامل.