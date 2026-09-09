/**
 * Nexus-Forge cytology — headless + GUI QuPath importer (v0.7.0+).
 *
 * CLI (headless):
 *   QuPath script -i <slide.svs> -s --args <qupath_export.json> qupath_enriched_visualizer.groovy
 *
 * GUI fallback:
 *   System.setProperty("nexus.json", "/path/to/qupath_export.json") then Run script.
 *
 * Contract: clinical[] uses Malignant (red), Immune (amber), Normal (green).
 */

import com.google.gson.Gson

import qupath.lib.color.ColorTools
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.RoiTools

def jsonFilePath = (binding.hasVariable("args") && args?.size() > 0)
    ? args[0]
    : System.getProperty("nexus.json")

if (jsonFilePath == null || jsonFilePath.toString().trim().isEmpty())
    throw new IllegalArgumentException(
            "Export JSON required. Use: QuPath script -i <slide> -s --args <qupath_export.json> qupath_enriched_visualizer.groovy")

/** Normalize Gson numbers / strings to double for QuPath measurements. */
def asDouble = { Object v ->
    if (v == null)
        return Double.NaN
    if (v instanceof Number)
        return ((Number) v).doubleValue()
    try {
        return Double.parseDouble(v.toString())
    } catch (NumberFormatException e) {
        return Double.NaN
    }
}

def gson = new Gson()
def root = gson.fromJson(new File(jsonFilePath).text, Map)

def cells = root["cells"] as List
def clinical = root["clinical"] as List
def spatial = root["spatial_features"] as List
def confidence = (root["confidence"] as List)

if (cells == null || clinical == null || spatial == null)
    throw new IllegalArgumentException("JSON must contain top-level keys: cells, clinical, spatial_features")

int n = cells.size()
if (clinical.size() != n || spatial.size() != n)
    throw new IllegalArgumentException(
            "Array length mismatch: cells=${n}, clinical=${clinical.size()}, spatial_features=${spatial.size()}")

def imageData = getCurrentImageData()
if (imageData == null)
    throw new IllegalStateException(
            "No image open. Use: QuPath script -i <slide> -s --args <json> qupath_enriched_visualizer.groovy")

def plane = ImagePlane.getDefaultPlane()
def hierarchy = imageData.getHierarchy()

// 3-class contract aligned with GNN clinical_label export.
def malClass  = PathClassFactory.getPathClass("Nexus: Malignant", ColorTools.packARGB(255, 229, 57, 53))
def immClass  = PathClassFactory.getPathClass("Nexus: Immune",   ColorTools.packARGB(255, 255, 179, 0))
def normClass = PathClassFactory.getPathClass("Nexus: Normal",   ColorTools.packARGB(255, 67, 160, 71))

def resolveClass = { String raw ->
    switch (raw?.trim()?.toLowerCase()) {
        case "malignant": return malClass
        case "immune":    return immClass
        default:          return normClass
    }
}

// Idempotent re-import: strip prior Nexus annotations before adding fresh set.
def prior = hierarchy.getAnnotationObjects().findAll {
    it.getPathClass()?.getName()?.startsWith("Nexus:")
}
prior.each { hierarchy.removeObject(it, false) }

def annotations = new ArrayList<>(n)

for (int i = 0; i < n; i++) {
    def ring = cells[i] as List
    def pts = new ArrayList<Point2>(ring.size())
    for (p in ring) {
        def xy = p as List
        pts.add(new Point2(asDouble(xy.get(0)), asDouble(xy.get(1))))
    }

    def roi = RoiTools.makePolygonROI(pts, plane)
    def pathClass = resolveClass(clinical[i]?.toString())
    def ann = PathObjects.createAnnotationObject(roi, pathClass)

    def sf = spatial[i] as Map
    def cid = sf.get("cell_id")
    ann.setName(cid != null ? "Cell_${cid}" : "Cell_${i}")

    def ml = ann.getMeasurementList()
    if (confidence != null && i < confidence.size())
        ml.putMeasurement("Nexus_Confidence", asDouble(confidence[i]))

    ml.putMeasurement("Area_Math", asDouble(sf.get("area")))
    ml.putMeasurement("Circularity", asDouble(sf.get("circularity")))
    ml.putMeasurement("KNN_Density", asDouble(sf.get("knn_density")))
    ml.putMeasurement("RBF_Risk_Score", asDouble(sf.get("rbf_risk_score")))
    ml.putMeasurement("Perimeter_Math", asDouble(sf.get("perimeter")))
    ml.putMeasurement("Eccentricity", asDouble(sf.get("eccentricity")))
    ml.putMeasurement("Tumor_Edge_Distance", asDouble(sf.get("distance_to_tumor_edge")))
    ml.close()

    annotations.add(ann)
}

hierarchy.addObjects(annotations)
fireHierarchyUpdate()

println("Nexus-Forge: imported ${annotations.size()} annotations from:")
println("  ${jsonFilePath}")
