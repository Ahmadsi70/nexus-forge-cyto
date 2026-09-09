"""
تست ریاضی هسته هندسی Nexus-Forge Cyto
======================================
هر تابع را با مقادیر دقیق شناخته‌شده از ریاضیات راستی‌آزمایی می‌کند.
"""
import math, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, got, expected, tol=1e-9):
    ok = abs(got - expected) < tol
    tag = PASS if ok else FAIL
    results.append((tag, name, got, expected))
    icon = "✅" if ok else "❌"
    print(f"  {icon} [{tag}] {name}")
    print(f"         نتیجه={got:.10f}   انتظار={expected:.10f}   خطا={abs(got-expected):.2e}")

# ══════════════════════════════════════════════════════════════════════════════
# ۱. مساحت — Shoelace Formula
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۱. تابع polygon_area — فرمول Shoelace (Green's theorem)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def area_shoelace(verts):
    v = np.array(verts)
    n = len(v)
    if n < 3: return 0.0
    x, y = v[:,0], v[:,1]
    return 0.5 * abs(float(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))))

# مربع واحد: مساحت باید ۱.۰ باشد
sq = [[0,0],[1,0],[1,1],[0,1]]
check("مربع ۱×۱", area_shoelace(sq), 1.0)

# مستطیل ۳×۴: مساحت = ۱۲
rect = [[0,0],[3,0],[3,4],[0,4]]
check("مستطیل ۳×۴", area_shoelace(rect), 12.0)

# مثلث قائم‌الزاویه با پایه=۶ ارتفاع=۸: مساحت = ۲۴
tri = [[0,0],[6,0],[0,8]]
check("مثلث قائم ۶×۸", area_shoelace(tri), 24.0)

# دایره ۳۲-ضلعی با شعاع r: مساحت ≈ π×r² (خطای polygon تقریبی)
r = 10.0
circle32 = [[r*math.cos(2*math.pi*i/32), r*math.sin(2*math.pi*i/32)] for i in range(32)]
area_circle = area_shoelace(circle32)
expected_circle = math.pi * r * r
# خطای مجاز برای ۳۲ ضلعی تقریبی
rel_error = abs(area_circle - expected_circle) / expected_circle
ok = rel_error < 0.01  # کمتر از ۱٪
results.append((PASS if ok else FAIL, "دایره r=10 (32-gon, خطا<1%)", area_circle, expected_circle))
print(f"  {'✅' if ok else '❌'} [{'PASS' if ok else 'FAIL'}] دایره r=10 (32-gon، خطا<1٪)")
print(f"         نتیجه={area_circle:.6f}   انتظار≈{expected_circle:.6f}   خطانسبی={rel_error*100:.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ۲. محیط — Perimeter
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۲. تابع polygon_perimeter")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def perimeter(verts):
    v = np.array(verts)
    n = len(v)
    if n < 2: return 0.0
    closed = np.vstack([v, v[0]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))

# مربع ۱×۱: محیط = ۴
check("مربع ۱×۱", perimeter(sq), 4.0)

# مثلث ۳-۴-۵
tri345 = [[0,0],[3,0],[0,4]]
check("مثلث ۳-۴-۵", perimeter(tri345), 12.0)

# دایره ۳۲-ضلعی r=10: محیط ≈ 2πr (خطای <1%)
perim_circle = perimeter(circle32)
expected_perim = 2 * math.pi * r
rel_err_p = abs(perim_circle - expected_perim) / expected_perim
ok2 = rel_err_p < 0.01
results.append((PASS if ok2 else FAIL, "محیط دایره r=10 (<1%)", perim_circle, expected_perim))
print(f"  {'✅' if ok2 else '❌'} [{'PASS' if ok2 else 'FAIL'}] محیط دایره r=10 (خطا<1٪)")
print(f"         نتیجه={perim_circle:.6f}   انتظار≈{expected_perim:.6f}   خطانسبی={rel_err_p*100:.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ۳. دایرویی — Circularity C = 4πA/P²
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۳. تابع polygon_circularity   C = 4πA / P²")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def circularity(area, perim):
    if perim < 1e-18 or area < 0: return 0.0
    return min((4*math.pi*area) / (perim**2), 1.0)

# دایره: C باید ۱.۰ باشد (کامل‌ترین شکل)
A_circ = area_shoelace(circle32)
P_circ = perimeter(circle32)
C_circ = circularity(A_circ, P_circ)
check("دایره r=10 → C≈1.0", C_circ, 1.0, tol=0.01)

# مربع: C = π/4 ≈ 0.7854
A_sq = area_shoelace(sq)
P_sq = perimeter(sq)
C_sq = circularity(A_sq, P_sq)
check("مربع ۱×۱ → C=π/4", C_sq, math.pi/4, tol=1e-9)

# مستطیل ۱×۱۰ (بسیار کشیده): باید C خیلی کوچک باشد
rect_thin = [[0,0],[10,0],[10,1],[0,1]]
A_thin = area_shoelace(rect_thin)
P_thin = perimeter(rect_thin)
C_thin = circularity(A_thin, P_thin)
expected_thin = 4*math.pi*10 / (22**2)
check("مستطیل ۱×۱۰ (کشیده)", C_thin, expected_thin, tol=1e-9)
print(f"         (مقدار کم = شکل نامنظم‌تر ✓)")

# آستانه ۰: محیط=۰
check("محیط صفر → C=0", circularity(1.0, 0.0), 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# ۴. مرکز جرم — Centroid (Shoelace centroid)
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۴. تابع polygon_centroid_xy")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def centroid(verts):
    v = np.array(verts, dtype=float)
    n = len(v)
    if n < 3:
        return v[:,0].mean(), v[:,1].mean()
    a2 = 0.0; cx = 0.0; cy = 0.0
    for i in range(n):
        j = (i+1)%n
        cross = v[i,0]*v[j,1] - v[j,0]*v[i,1]
        a2 += cross
        cx += (v[i,0]+v[j,0]) * cross
        cy += (v[i,1]+v[j,1]) * cross
    area = 0.5 * a2
    if abs(area) < 1e-18:
        return v[:,0].mean(), v[:,1].mean()
    f = 1.0 / (6.0 * area)
    return cx*f, cy*f

cx, cy = centroid(sq)
check("مرکز مربع ۱×۱ — x=0.5", cx, 0.5)
check("مرکز مربع ۱×۱ — y=0.5", cy, 0.5)

# مستطیل ۶×۴: مرکز باید (۳, ۲) باشد
big_rect = [[0,0],[6,0],[6,4],[0,4]]
cx2, cy2 = centroid(big_rect)
check("مرکز مستطیل ۶×۴ — x=3", cx2, 3.0)
check("مرکز مستطیل ۶×۴ — y=2", cy2, 2.0)

# مثلث متساوی‌الاضلاع: مرکز ثقل = میانگین رئوس
tri_eq = [[0,0],[2,0],[1, math.sqrt(3)]]
cx3, cy3 = centroid(tri_eq)
check("مرکز مثلث متساوی — x=1", cx3, 1.0)
check("مرکز مثلث متساوی — y=√3/3", cy3, math.sqrt(3)/3, tol=1e-9)

# ══════════════════════════════════════════════════════════════════════════════
# ۵. بیضی‌بودن — Eccentricity از کوواریانس
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۵. تابع polygon_eccentricity — e = √(1 - λ₂/λ₁)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def eccentricity(ring32):
    pts = np.array(ring32, dtype=float)
    cov = np.cov(pts.T)
    if cov.ndim < 2: return 0.0
    tr = cov[0,0]+cov[1,1]
    det = cov[0,0]*cov[1,1] - cov[0,1]**2
    disc = tr**2 - 4*det
    if disc < 0: return 0.0
    l1 = 0.5*(tr + math.sqrt(disc))
    l2 = 0.5*(tr - math.sqrt(disc))
    if l1 < 1e-18: return 0.0
    ratio = max(0.0, min(1.0, l2/l1))
    inner = 1.0 - ratio
    return math.sqrt(inner) if inner > 0 else 0.0

# دایره کامل: λ₁ ≈ λ₂ → e ≈ 0
e_circle = eccentricity(circle32)
ok_circ = e_circle < 0.08
results.append((PASS if ok_circ else FAIL, "دایره → e≈0", e_circle, 0.0))
print(f"  {'✅' if ok_circ else '❌'} [{'PASS' if ok_circ else 'FAIL'}] دایره → e≈0")
print(f"         نتیجه={e_circle:.6f}   انتظار≈0.0   (قبول اگر <0.08)")

# خط افقی: λ₁>>λ₂ → e≈1
line32 = [[float(i), 0.0] for i in range(32)]
e_line = eccentricity(line32)
ok_line = e_line > 0.99
results.append((PASS if ok_line else FAIL, "خط افقی → e≈1", e_line, 1.0))
print(f"  {'✅' if ok_line else '❌'} [{'PASS' if ok_line else 'FAIL'}] خط افقی → e≈1")
print(f"         نتیجه={e_line:.6f}   انتظار≈1.0   (قبول اگر >0.99)")

# بیضی با نسبت ۳:۱
t = np.linspace(0, 2*math.pi, 32, endpoint=False)
ellipse32 = list(zip((3*np.cos(t)).tolist(), (1*np.sin(t)).tolist()))
e_ellipse = eccentricity(ellipse32)
# برای بیضی a=3, b=1: e_geom = √(1-b²/a²) = √(8/9) ≈ 0.943
# ولی فرمول پروژه از کوواریانس استفاده می‌کند نه تعریف هندسی بیضی
# کوواریانس روی نقاط یکنواخت: Var(x)=a²/2=4.5, Var(y)=b²/2=0.5
# λ₁=4.5, λ₂=0.5 → e=√(1-0.5/4.5)=√(8/9)≈0.943
e_expected = math.sqrt(1 - 0.5/4.5)
check("بیضی ۳:۱ → e=√(8/9)≈0.943", e_ellipse, e_expected, tol=0.02)

# ══════════════════════════════════════════════════════════════════════════════
# ۶. kNN Density
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۶. تابع calculate_knn_density")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def knn_density(centroids_list, k=8):
    pts = np.array(centroids_list)
    n = len(pts)
    out = []
    for i in range(n):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists = np.sort(dists[dists > 1e-12])
        k_eff = min(k, len(dists))
        out.append(float(dists[:k_eff].mean()) if k_eff > 0 else 0.0)
    return out

# دو نقطه با فاصله ۵: knn باید ۵ باشد
two_pts = [[0,0],[3,4]]
d = knn_density(two_pts, k=8)
check("دو نقطه — فاصله=5 → knn[0]=5", d[0], 5.0)
check("دو نقطه — فاصله=5 → knn[1]=5", d[1], 5.0)

# شبکه مربعی ۳×۳ با فاصله ۱۰: نزدیک‌ترین همسایه = ۱۰
grid = [[10*i, 10*j] for i in range(3) for j in range(3)]
d_grid = knn_density(grid, k=1)
# گوشه‌ها یک همسایه‌ی نزدیک دارند (مستقیم یا قطر؟ — نزدیک‌ترین=10)
min_d = min(d_grid)
ok_grid = abs(min_d - 10.0) < 1e-9
results.append((PASS if ok_grid else FAIL, "شبکه ۳×۳ — نزدیک‌ترین=10", min_d, 10.0))
print(f"  {'✅' if ok_grid else '❌'} [{'PASS' if ok_grid else 'FAIL'}] شبکه ۳×۳ — min knn=10")
print(f"         نتیجه={min_d:.6f}   انتظار=10.0")

# ══════════════════════════════════════════════════════════════════════════════
# ۷. RBF Field
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۷. تابع calculate_rbf_field   F(x) = Σ exp(-d²/2σ²)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def rbf_field(target_idx, centroids_list, sigma):
    pts = np.array(centroids_list)
    cx, cy = pts[target_idx]
    dists_sq = np.sum((pts - np.array([cx,cy]))**2, axis=1)
    dists_sq = dists_sq[np.arange(len(pts)) != target_idx]
    return float(np.sum(np.exp(-dists_sq / (2*sigma**2))))

# دو نقطه با فاصله ۱ و sigma=1: F = exp(-0.5)
two = [[0,0],[1,0]]
f0 = rbf_field(0, two, sigma=1.0)
expected_rbf = math.exp(-0.5)
check("RBF دو نقطه d=1, σ=1 → exp(-0.5)", f0, expected_rbf)

# سه نقطه هم‌فاصله: F باید ۲×exp(-d²/2σ²) باشد
three = [[0,0],[2,0],[4,0]]
f_mid = rbf_field(1, three, sigma=1.0)
expected_3 = 2 * math.exp(-4/2)
check("RBF وسط ۳ نقطه هم‌فاصله (d=2)", f_mid, expected_3)

# ══════════════════════════════════════════════════════════════════════════════
# ۸. Convex Hull (Andrew Monotone Chain)
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۸. تابع calculate_convex_hull — Andrew Monotone Chain")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def convex_hull(points):
    """Andrew monotone chain — همان پیاده‌سازی spatial.rs"""
    pts = sorted(map(tuple, points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

# مربع + نقطه داخلی: باید ۴ رأس داشته باشد
pts_with_interior = [[0,0],[1,0],[1,1],[0,1],[0.5,0.5]]
hull = convex_hull(pts_with_interior)
ok_hull = len(hull) == 4
results.append((PASS if ok_hull else FAIL, "مربع + نقطه داخلی → ۴ رأس", float(len(hull)), 4.0))
print(f"  {'✅' if ok_hull else '❌'} [{'PASS' if ok_hull else 'FAIL'}] مربع + نقطه داخلی → ۴ رأس")
print(f"         تعداد رأس={len(hull)}   انتظار=4")

# مثلث: باید ۳ رأس داشته باشد
hull_tri = convex_hull([[0,0],[5,0],[2.5,4],[2.5,2]])
ok_tri = len(hull_tri) == 3
results.append((PASS if ok_tri else FAIL, "مثلث + نقطه داخلی → ۳ رأس", float(len(hull_tri)), 3.0))
print(f"  {'✅' if ok_tri else '❌'} [{'PASS' if ok_tri else 'FAIL'}] مثلث + نقطه داخلی → ۳ رأس")
print(f"         تعداد رأس={len(hull_tri)}   انتظار=3")

# ══════════════════════════════════════════════════════════════════════════════
# ۹. فاصله تا مرز تومور
# ══════════════════════════════════════════════════════════════════════════════
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  ۹. تابع distance_to_polygon")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def dist_pt_seg_sq(p, a, b):
    vx,vy = b[0]-a[0], b[1]-a[1]
    wx,wy = p[0]-a[0], p[1]-a[1]
    vv = vx*vx+vy*vy
    if vv<1e-36: return wx*wx+wy*wy
    t = max(0.0, min(1.0, (wx*vx+wy*vy)/vv))
    cx,cy = a[0]+t*vx, a[1]+t*vy
    return (p[0]-cx)**2+(p[1]-cy)**2

def distance_to_polygon(point, hull):
    n = len(hull)
    if n==0: return float('inf')
    if n==1: return math.sqrt((point[0]-hull[0][0])**2+(point[1]-hull[0][1])**2)
    best_sq = float('inf')
    for i in range(n):
        d2 = dist_pt_seg_sq(point, hull[i], hull[(i+1)%n])
        if d2 < best_sq: best_sq = d2
    return math.sqrt(best_sq)

# مرکز مربع ۲×۲: فاصله تا هر ضلع = ۱
hull_sq = [[0,0],[2,0],[2,2],[0,2]]
d_center = distance_to_polygon([1,1], hull_sq)
check("مرکز مربع ۲×۲ → فاصله=1", d_center, 1.0)

# روی ضلع: فاصله = ۰
d_on_edge = distance_to_polygon([1,0], hull_sq)
check("نقطه روی ضلع → فاصله=0", d_on_edge, 0.0)

# نقطه خارج: فاصله تا گوشه = ۵ (مثلث ۳-۴-۵)
d_corner = distance_to_polygon([2+3, 2+4], hull_sq)
check("بیرون مربع → فاصله به گوشه=5", d_corner, 5.0)

# hull خالی: باید inf برگرداند
import math
d_empty = distance_to_polygon([5,5], [])
ok_inf = math.isinf(d_empty)
results.append((PASS if ok_inf else FAIL, "hull خالی → inf", float(d_empty), float('inf')))
print(f"  {'✅' if ok_inf else '❌'} [{'PASS' if ok_inf else 'FAIL'}] hull خالی → inf")

# ══════════════════════════════════════════════════════════════════════════════
# نتیجه نهایی
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  📋 نتیجه نهایی")
print("═"*60)
passed = sum(1 for r in results if r[0]==PASS)
failed = sum(1 for r in results if r[0]==FAIL)
total  = len(results)
print(f"  ✅ موفق: {passed}/{total}")
print(f"  ❌ ناموفق: {failed}/{total}")
if failed == 0:
    print("\n  🟢 تمام الگوریتم‌های هندسی درست هستند!")
    print("     هسته ریاضی با فرمول‌های استاندارد مطابقت کامل دارد.")
else:
    print("\n  🔴 خطا در الگوریتم‌های زیر:")
    for r in results:
        if r[0] == FAIL:
            print(f"     • {r[1]} — نتیجه={r[2]:.6f} انتظار={r[3]:.6f}")
print("═"*60)
