/**
 * Nexus-Forge Cyto — Enhanced QuPath Visualizer (v0.3.0)
 *
 * === Usage ===
 *
 * CLI (headless):
 *   QuPath script -i <slide.svs> -s --args <qupath_export.json> qupath_enriched_visualizer.groovy
 *
 * GUI (interactive):
 *   Drag & drop into QuPath → Run for project  (auto-detects JSON in project dir)
 *   Or set system property:  System.setProperty("nexus.json", "/path/to/qupath_export.json")
 *
 * === Visualization Modes ===
 *   Mode 1 — Clinical Classification:  Malignant (red), Immune (amber), Normal (green)
 *   Mode 2 — RBF Risk Heatmap:        Blue (low) → Yellow (medium) → Red (high risk)
 *   Mode 3 — Distance-to-Edge:        Far (blue) → Near tumor edge (red)
 *
 * === Features ===
 *   - Auto-detects enriched JSON in QuPath project directory
 *   - Legend panel as QuPath annotation overlay
 *   - Summary statistics printed to console
 *   - Per-cell measurements (Area, Circularity, KNN Density, RBF Risk, Eccentricity,
 *     Tumor Edge Distance, Perimeter)
 *   - Idempotent: strips prior Nexus annotations before re-import
 */

import com.google.gson.Gson

import qupath.lib.color.ColorTools
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RoiTools

import java.awt.Color

// ── Configuration ────────────────────────────────────────────────────────────

// Visualization mode: "clinical" | "heatmap" | "distance"
def VIS_MODE = System.getProperty("nexus.vis.mode", "clinical")

// Opacity for annotation overlays (0.0–1.0)
def OVERLAY_OPACITY = Double.parseDouble(
    System.getProperty("nexus.vis.opacity", "0.55")
)

// Show legend as QuPath annotation
def SHOW_LEGEND = Boolean.parseBoolean(
    System.getProperty("nexus.vis.legend", "true")
)

// ── JSON Resolution (auto-detect or explicit) ────────────────────────────────

def jsonFilePath = resolveJsonPath()
if (jsonFilePath == null || jsonFilePath.toString().trim().isEmpty()) {
    throw new IllegalArgumentException(
        "No enriched JSON found.\n" +
        "  Explicit: QuPath script -i <slide> -s --args <qupath_export.json> qupath_enriched_visualizer.groovy\n" +
        "  System prop: -Dnexus.json=/path/to/qupath_export.json\n" +
        "  Auto-detect: place *_enriched.json or qupath_export.json in the QuPath project directory"
    )
}

println("Nexus-Forge Visualizer v0.3.0")
println("  JSON:       ${jsonFilePath}")
println("  Mode:       ${VIS_MODE}")
println("  Legend:     ${SHOW_LEGEND}")
println("  Opacity:    ${OVERLAY_OPACITY}")

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Normalize Gson numbers / strings to double for QuPath measurements. */
def asDouble = { Object v ->
    if (v == null) return Double.NaN
    if (v instanceof Number) return ((Number) v).doubleValue()
    try { return Double.parseDouble(v.toString()) }
    catch (NumberFormatException e) { return Double.NaN }
}

/** Clamp a double to [0, 1] for color interpolation. */
def clamp01 = { double v -> Math.max(0.0, Math.min(1.0, v)) }

/** Linear interpolate between two ARGB-packed ints by t ∈ [0, 1]. */
def lerpColor = { int c0, int c1, double t ->
    double tt = clamp01(t)
    int a = (int) (((c0 >> 24) & 0xFF) * (1 - tt) + ((c1 >> 24) & 0xFF) * tt)
    int r = (int) (((c0 >> 16) & 0xFF) * (1 - tt) + ((c1 >> 16) & 0xFF) * tt)
    int g = (int) (((c0 >> 8)  & 0xFF) * (1 - tt) + ((c1 >> 8)  & 0xFF) * tt)
    int b = (int) (( c0        & 0xFF) * (1 - tt) + ( c1        & 0xFF) * tt)
    return ColorTools.packARGB(a, r, g, b)
}

/** Resolve the enriched JSON path from args, system property, or auto-detect. */
def resolveJsonPath() {
    // 1. CLI argument
    if (binding.hasVariable("args") && args?.size() > 0) {
        def f = new File(args[0])
        if (f.exists()) return f.absolutePath
    }
    // 2. System property
    def sysProp = System.getProperty("nexus.json")
    if (sysProp != null) {
        def f = new File(sysProp)
        if (f.exists()) return f.absolutePath
    }
    // 3. Auto-detect in QuPath project directory
    try {
        def imgData = getCurrentImageData()
        if (imgData != null) {
            def project = imgData.getProject()
            if (project != null) {
                def projectDir = project.getDirectory().toFile()
                if (projectDir != null && projectDir.exists()) {
                    // Look for *_enriched.json or qupath_export.json
                    def candidates = projectDir.listFiles().findAll { f ->
                        f.name.endsWith(".json") &&
                        (f.name.contains("enriched") || f.name.contains("qupath_export"))
                    }
                    if (candidates) {
                        // Prefer newest
                        candidates.sort { a, b -> b.lastModified() <=> a.lastModified() }
                        println("Auto-detected: ${candidates[0].name}")
                        return candidates[0].absolutePath
                    }
                }
            }
        }
    } catch (Exception e) {
        // GUI context not available (headless mode without args)
    }
    return null
}

// ── Load JSON ────────────────────────────────────────────────────────────────

def gson = new Gson()
def root = gson.fromJson(new File(jsonFilePath).text, Map)

def cells     = root["cells"] as List
def clinical  = root["clinical"] as List
def spatial   = root["spatial_features"] as List
def confidence = (root["confidence"] as List)

if (cells == null || clinical == null || spatial == null) {
    throw new IllegalArgumentException(
        "JSON must contain top-level keys: cells, clinical, spatial_features"
    )
}

int n = cells.size()
if (clinical.size() != n || spatial.size() != n) {
    throw new IllegalArgumentException(
        "Array length mismatch: cells=${n}, clinical=${clinical.size()}, spatial_features=${spatial.size()}"
    )
}

// ── Compute value ranges for heatmap/distance normalisation ──────────────────

double rbfMin = Double.MAX_VALUE, rbfMax = -Double.MAX_VALUE
double edgeMin = Double.MAX_VALUE, edgeMax = -Double.MAX_VALUE
double areaMin = Double.MAX_VALUE, areaMax = -Double.MAX_VALUE
double confMin = Double.MAX_VALUE, confMax = -Double.MAX_VALUE

for (int i = 0; i < n; i++) {
    def sf = spatial[i] as Map
    double rbf  = asDouble(sf.get("rbf_risk_score"))
    double edge = asDouble(sf.get("distance_to_tumor_edge"))
    double area = asDouble(sf.get("area"))
    double conf = (confidence != null && i < confidence.size()) ? asDouble(confidence[i]) : Double.NaN

    if (!Double.isNaN(rbf))  { rbfMin  = Math.min(rbfMin,  rbf);  rbfMax  = Math.max(rbfMax,  rbf)  }
    if (!Double.isNaN(edge)) { edgeMin = Math.min(edgeMin, edge); edgeMax = Math.max(edgeMax, edge) }
    if (!Double.isNaN(area)) { areaMin = Math.min(areaMin, area); areaMax = Math.max(areaMax, area) }
    if (!Double.isNaN(conf)) { confMin = Math.min(confMin, conf); confMax = Math.max(confMax, conf) }
}

// Fallback ranges if all NaN or single-valued
if (rbfMax <= rbfMin)  { rbfMin = 0.0;  rbfMax = 1.0  }
if (edgeMax <= edgeMin) { edgeMin = 0.0; edgeMax = 100.0 }
if (areaMax <= areaMin) { areaMin = 0.0; areaMax = 1.0  }

// ── Color palettes ───────────────────────────────────────────────────────────

// RBF risk: blue (low) → cyan → yellow → orange → red (high)
def rbfLow  = ColorTools.packARGB(192, 33,  150, 243)  // blue
def rbfMid  = ColorTools.packARGB(192, 255, 235, 59)   // yellow
def rbfHigh = ColorTools.packARGB(192, 229, 57,  53)   // red

// Distance-to-edge: far (blue) → mid (green) → near edge (red)
def edgeFar  = ColorTools.packARGB(192, 33,  150, 243) // blue
def edgeMid  = ColorTools.packARGB(192, 76,  175, 80)  // green
def edgeNear = ColorTools.packARGB(192, 229, 57,  53)  // red

// Clinical classification palette
def malColor  = ColorTools.packARGB((int)(OVERLAY_OPACITY * 255), 229, 57,  53)
def immColor  = ColorTools.packARGB((int)(OVERLAY_OPACITY * 255), 255, 179, 0)
def normColor = ColorTools.packARGB((int)(OVERLAY_OPACITY * 255), 67,  160, 71)
def unkColor  = ColorTools.packARGB((int)(OVERLAY_OPACITY * 255), 158, 158, 158)

// ── PathClass factories ──────────────────────────────────────────────────────

def malClass  = PathClassFactory.getPathClass("Nexus: Malignant", malColor)
def immClass  = PathClassFactory.getPathClass("Nexus: Immune",   immColor)
def normClass = PathClassFactory.getPathClass("Nexus: Normal",   normColor)
def unkClass  = PathClassFactory.getPathClass("Nexus: Unknown",  unkColor)

// Dynamic PathClass for heatmap/distance modes (we create them per-color-bin)
def heatClasses = [:]

def resolveClinicalClass = { String raw ->
    switch (raw?.trim()?.toLowerCase()) {
        case "malignant": return malClass
        case "immune":    return immClass
        default:          return normClass
    }
}

/** Resolve a color and PathClass for the given visualization mode and index. */
def resolveVisClass = { int idx, String mode ->
    def sf = spatial[idx] as Map
    switch (mode) {
        case "heatmap":
            double rbf = asDouble(sf.get("rbf_risk_score"))
            if (Double.isNaN(rbf)) return unkClass
            double t = clamp01((rbf - rbfMin) / (rbfMax - rbfMin))
            int color
            if (t < 0.5) {
                color = lerpColor(rbfLow, rbfMid, t * 2.0)
            } else {
                color = lerpColor(rbfMid, rbfHigh, (t - 0.5) * 2.0)
            }
            // Cache PathClass by integer color for performance
            String key = "heat_${color}"
            if (!heatClasses.containsKey(key)) {
                heatClasses[key] = PathClassFactory.getPathClass(
                    "Nexus: RBF ${String.format('%.2f', t)}", color
                )
            }
            return heatClasses[key]

        case "distance":
            double edge = asDouble(sf.get("distance_to_tumor_edge"))
            if (Double.isNaN(edge)) return unkClass
            double t = clamp01((edge - edgeMin) / (edgeMax - edgeMin))
            int color
            if (t < 0.5) {
                color = lerpColor(edgeNear, edgeMid, t * 2.0)
            } else {
                color = lerpColor(edgeMid, edgeFar, (t - 0.5) * 2.0)
            }
            String key = "dist_${color}"
            if (!heatClasses.containsKey(key)) {
                heatClasses[key] = PathClassFactory.getPathClass(
                    "Nexus: Edge ${String.format('%.1f', edge)}µm", color
                )
            }
            return heatClasses[key]

        default: // "clinical"
            return resolveClinicalClass(clinical[idx]?.toString())
    }
}

// ── QuPath setup ─────────────────────────────────────────────────────────────

def imageData = getCurrentImageData()
if (imageData == null) {
    throw new IllegalStateException(
        "No image open. Use: QuPath script -i <slide> -s --args <json> qupath_enriched_visualizer.groovy"
    )
}

def plane = ImagePlane.getDefaultPlane()
def hierarchy = imageData.getHierarchy()

// Idempotent: strip prior Nexus annotations before adding fresh set
def prior = hierarchy.getAnnotationObjects().findAll {
    it.getPathClass()?.getName()?.startsWith("Nexus:")
}
prior.each { hierarchy.removeObject(it, false) }

// ── Create annotations ───────────────────────────────────────────────────────

def annotations = new ArrayList<>(n)

for (int i = 0; i < n; i++) {
    def ring = cells[i] as List
    def pts = new ArrayList<Point2>(ring.size())
    for (p in ring) {
        def xy = p as List
        pts.add(new Point2(asDouble(xy.get(0)), asDouble(xy.get(1))))
    }

    def roi = RoiTools.makePolygonROI(pts, plane)
    def pathClass = resolveVisClass(i, VIS_MODE)
    def ann = PathObjects.createAnnotationObject(roi, pathClass)

    def sf = spatial[i] as Map
    def cid = sf.get("cell_id")
    ann.setName(cid != null ? "Cell_${cid}" : "Cell_${i}")

    // Measurements
    def ml = ann.getMeasurementList()
    if (confidence != null && i < confidence.size())
        ml.putMeasurement("Nexus_Confidence", asDouble(confidence[i]))

    ml.putMeasurement("Area_Math",         asDouble(sf.get("area")))
    ml.putMeasurement("Circularity",       asDouble(sf.get("circularity")))
    ml.putMeasurement("KNN_Density",       asDouble(sf.get("knn_density")))
    ml.putMeasurement("RBF_Risk_Score",    asDouble(sf.get("rbf_risk_score")))
    ml.putMeasurement("Perimeter_Math",    asDouble(sf.get("perimeter")))
    ml.putMeasurement("Eccentricity",      asDouble(sf.get("eccentricity")))
    ml.putMeasurement("Tumor_Edge_Dist",   asDouble(sf.get("distance_to_tumor_edge")))
    ml.close()

    annotations.add(ann)
}

hierarchy.addObjects(annotations)
fireHierarchyUpdate()

println("Nexus-Forge: imported ${annotations.size()} annotations from:")
println("  ${jsonFilePath}")

// ── Legend annotation ────────────────────────────────────────────────────────

if (SHOW_LEGEND) {
    addLegend(hierarchy, plane, imageData)
}

/**
 * Create a legend as a rectangle annotation with descriptive text
 * positioned in the top-left corner of the image.
 */
def addLegend(hierarchy, plane, imageData) {
    def server = imageData.getServer()
    // Position legend in top-left, scaled to image dimensions
    double imgW = server.getWidth()
    double imgH = server.getHeight()

    // Legend box dimensions (in pixel coordinates)
    double boxX = 20.0
    double boxY = 20.0
    double boxW = 280.0
    double boxH = 140.0

    // Scale to downsample if needed
    double downsample = server.getDownsampleForResolution(0)
    boxW *= downsample
    boxH *= downsample

    // Build legend text
    def lines = []
    lines.add("Nexus-Forge Cyto v0.3.0")
    lines.add("─" * 28)
    lines.add("Mode: ${VIS_MODE.toUpperCase()}")

    switch (VIS_MODE) {
        case "clinical":
            lines.add("■ Malignant  ■ Immune  ■ Normal")
            break
        case "heatmap":
            lines.add("RBF Risk:  ■ Low  ■ Mid  ■ High")
            lines.add("Range: [${String.format('%.3f', rbfMin)}, ${String.format('%.3f', rbfMax)}]")
            break
        case "distance":
            lines.add("Edge Dist:  ■ Near  ■ Mid  ■ Far")
            lines.add("Range: [${String.format('%.1f', edgeMin)}, ${String.format('%.1f', edgeMax)}] µm")
            break
    }

    lines.add("Cells: ${n}")
    lines.add("─" * 28)
    lines.add("Modes: -Dnexus.vis.mode=clinical|heatmap|distance")

    // Create legend text as a single string for the annotation name
    def legendText = lines.join("\n")

    // Create a rectangle annotation for the legend background
    def legendRectPts = [
        new Point2(boxX, boxY),
        new Point2(boxX + boxW, boxY),
        new Point2(boxX + boxW, boxY + boxH),
        new Point2(boxX, boxY + boxH),
    ]
    def legendRoi = RoiTools.makePolygonROI(legendRectPts, plane)
    def legendBgColor = ColorTools.packARGB(220, 255, 255, 255)
    def legendClass = PathClassFactory.getPathClass("Nexus: Legend", legendBgColor)
    def legendAnn = PathObjects.createAnnotationObject(legendRoi, legendClass)
    legendAnn.setName(legendText)
    hierarchy.addObject(legendAnn)
}

// ── Summary statistics ───────────────────────────────────────────────────────

println("")
println("=" * 60)
println("  Nexus-Forge Cyto — Summary Statistics")
println("=" * 60)
println("  Total cells:       ${n}")

// Count clinical classes
int nMal = clinical.count { it?.toString()?.trim()?.toLowerCase() == "malignant" }
int nImm = clinical.count { it?.toString()?.trim()?.toLowerCase() == "immune" }
int nNorm = n - nMal - nImm
println("  Malignant:         ${nMal}  (${String.format('%.1f', 100.0*nMal/n)}%)")
println("  Immune:            ${nImm}  (${String.format('%.1f', 100.0*nImm/n)}%)")
println("  Normal:            ${nNorm}  (${String.format('%.1f', 100.0*nNorm/n)}%)")

// Spatial metrics aggregates
def areas = spatial.collect { asDouble((it as Map).get("area")) }.findAll { !Double.isNaN(it) }
def rbfs  = spatial.collect { asDouble((it as Map).get("rbf_risk_score")) }.findAll { !Double.isNaN(it) }
def edges = spatial.collect { asDouble((it as Map).get("distance_to_tumor_edge")) }.findAll { !Double.isNaN(it) }
def eccs  = spatial.collect { asDouble((it as Map).get("eccentricity")) }.findAll { !Double.isNaN(it) }

if (areas) println("  Mean Area:         ${String.format('%.2f', areas.sum() / areas.size())} px²")
if (rbfs)  println("  Mean RBF Risk:     ${String.format('%.4f', rbfs.sum() / rbfs.size())}")
if (edges) println("  Mean Edge Dist:    ${String.format('%.2f', edges.sum() / edges.size())} µm")
if (eccs)  println("  Mean Eccentricity: ${String.format('%.3f', eccs.sum() / eccs.size())}")

println("=" * 60)
println("  Visualization mode: ${VIS_MODE}")
println("  Change mode via:    -Dnexus.vis.mode=clinical|heatmap|distance")
println("=" * 60)