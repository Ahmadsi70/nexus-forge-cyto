/**
 * Nexus-Forge Cyto — QuPath Extension
 *
 * One-click "Analyze with Nexus-Forge" integration for QuPath >= 0.5.0.
 *
 * INSTALLATION:
 *   1. Copy this entire `qupath-extension/` directory into your QuPath
 *      extensions folder (typically `~/QuPath/extensions/` or
 *      `C:\\Users\\<user>\\QuPath\\extensions\\`).
 *   2. Set the `NEXUS_FORGE_HOME` environment variable to point at your
 *      nexus-forge-cyto checkout, or edit `getNexusHome()` below.
 *   3. Restart QuPath. A new menu item "Extensions → Nexus-Forge →
 *      Analyze with Nexus-Forge" will appear.
 *
 * REQUIREMENTS:
 *   - QuPath >= 0.5.0
 *   - Java 17+
 *   - Python >= 3.10 with nexus-forge-cyto installed (pip install -e .)
 *     OR a compiled Rust binary `nexus-forge-cyto-batch` on PATH
 *
 * LICENSE: MIT OR Apache-2.0
 */

import qupath.extensions.QuPathExtension
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.extensions.QuPathExtension
import qupath.lib.gui.tools.MenuTools
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RoiTools
import qupath.lib.roi.ROIs
import qupath.lib.scripting.QPEx

import javax.swing.JOptionPane
import javax.swing.SwingWorker
import java.awt.Color
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.TimeUnit


class NexusForgeExtension implements QuPathExtension {

    // ── Extension Metadata ──────────────────────────────────────────────

    @Override
    String getName() {
        return "Nexus-Forge Cyto"
    }

    @Override
    String getDescription() {
        return "Explainable geometry analysis for digital pathology. " +
               "Adds transparent morphometrics, spatial topology (kNN/RBF), " +
               "and auditable malignancy scoring to QuPath annotations."
    }

    @Override
    String getVersion() {
        return "0.2.0"
    }

    // ── Installation ────────────────────────────────────────────────────

    @Override
    void installExtension(QuPathGUI qupath, Map<String, Object> args) {
        // Add menu item under Extensions
        def menu = qupath.getMenu("Extensions", true)
        MenuTools.addMenuItems(menu, [
            MenuTools.createMenuItem("Nexus-Forge", null,
                MenuTools.createMenuItem("Analyze with Nexus-Forge",
                    "Analyze selected annotations with Nexus-Forge Cyto",
                    { -> runNexusAnalysis(qupath) }
                )
            )
        ])

        println "[Nexus-Forge] Extension v${getVersion()} installed."
        println "[Nexus-Forge] Use Extensions → Nexus-Forge → Analyze with Nexus-Forge"
    }

    // ── Helper: resolve Nexus-Forge home ────────────────────────────────

    static Path getNexusHome() {
        // 1. Environment variable
        def envHome = System.getenv("NEXUS_FORGE_HOME")
        if (envHome && Files.isDirectory(Paths.get(envHome))) {
            return Paths.get(envHome)
        }

        // 2. Look next to QuPath extensions
        def extDir = QuPathGUI.getExtensionClassLoader()
        if (extDir) {
            // Not directly accessible — fall back to common locations
        }

        // 3. Common locations
        def candidates = [
            Paths.get(System.getProperty("user.home"), "nexus-forge-cyto"),
            Paths.get(System.getProperty("user.home"), "cancer_project"),
            Paths.get("C:", "Users", System.getProperty("user.name"), "cancer_project"),
        ]
        for (cand in candidates) {
            if (Files.isDirectory(cand) && Files.exists(cand.resolve("Cargo.toml"))) {
                return cand
            }
        }
        return null
    }

    // ── Main Analysis Entry Point ───────────────────────────────────────

    static void runNexusAnalysis(QuPathGUI qupath) {
        def imageData = qupath.getImageData()
        if (!imageData) {
            JOptionPane.showMessageDialog(qupath.getStage(),
                "No image open. Please open a slide first.",
                "Nexus-Forge", JOptionPane.WARNING_MESSAGE)
            return
        }

        def annotations = imageData.getHierarchy().getAnnotationObjects()
        def cellAnnotations = annotations.findAll {
            it.getROI().isArea() && it.getPathClass() != null
        }

        if (cellAnnotations.isEmpty()) {
            JOptionPane.showMessageDialog(qupath.getStage(),
                "No annotation objects found. Please create cell annotations first\n" +
                "(e.g., using QuPath's cell detection or StarDist extension).",
                "Nexus-Forge", JOptionPane.WARNING_MESSAGE)
            return
        }

        def nexusHome = getNexusHome()
        if (!nexusHome) {
            JOptionPane.showMessageDialog(qupath.getStage(),
                "Nexus-Forge not found.\n\n" +
                "Set the NEXUS_FORGE_HOME environment variable to point at\n" +
                "your nexus-forge-cyto checkout, or install via pip:\n\n" +
                "    pip install nexus-forge-cyto",
                "Nexus-Forge — Not Found", JOptionPane.ERROR_MESSAGE)
            return
        }

        // Run analysis in background with progress dialog
        def worker = new SwingWorker<Path, Void>() {
            @Override
            protected Path doInBackground() throws Exception {
                return analyzeAnnotations(imageData, cellAnnotations, nexusHome)
            }

            @Override
            protected void done() {
                try {
                    def resultPath = get()
                    if (resultPath) {
                        loadEnrichedResults(qupath, imageData, resultPath)
                        JOptionPane.showMessageDialog(qupath.getStage(),
                            "Analysis complete!\n\n" +
                            "Results loaded into hierarchy.\n" +
                            "Red = Malignant, Blue = Normal\n" +
                            "Check Measurements tab for per-cell metrics.",
                            "Nexus-Forge — Complete", JOptionPane.INFORMATION_MESSAGE)
                    }
                } catch (Exception e) {
                    JOptionPane.showMessageDialog(qupath.getStage(),
                        "Analysis failed: ${e.getMessage()}",
                        "Nexus-Forge — Error", JOptionPane.ERROR_MESSAGE)
                    e.printStackTrace()
                }
            }
        }
        worker.execute()
    }

    // ── Export annotations to GeoJSON, call Nexus-Forge, get enriched ────

    static Path analyzeAnnotations(imageData, annotations, Path nexusHome) {
        // 1. Export selected annotations as GeoJSON
        def tempDir = Files.createTempDirectory("nexus-forge-")
        def geojsonPath = tempDir.resolve("annotations.geojson")

        // Build GeoJSON FeatureCollection from annotations
        def features = []
        annotations.eachWithIndex { ann, idx ->
            def roi = ann.getROI()
            def points = roi.getAllPoints()
            if (points.size() < 3) return  // skip degenerate

            def coords = points.collect { [it.getX(), it.getY()] }
            // Close the ring
            if (coords[0] != coords[-1]) {
                coords.add(coords[0])
            }

            features.add([
                type: "Feature",
                id: idx,
                geometry: [
                    type: "Polygon",
                    coordinates: [coords]
                ],
                properties: [
                    classification: ann.getPathClass()?.toString() ?: "Unknown",
                    object_type: "annotation"
                ]
            ])
        }

        def geojson = [
            type: "FeatureCollection",
            features: features
        ]

        geojsonPath.toFile().withWriter("UTF-8") { writer ->
            new groovy.json.JsonBuilder(geojson).writeTo(writer)
        }

        // 2. Call Nexus-Forge via Python or Rust CLI
        def enrichedPath = tempDir.resolve("enriched.json")

        // Try Python first, then Rust CLI
        def pythonScript = nexusHome.resolve(
            "nexus-forge-cyto/services/nexus_core/coord_adapter.py")
        def rustBinary = findRustBinary(nexusHome)

        if (Files.exists(pythonScript)) {
            runPythonAnalysis(pythonScript, geojsonPath, enrichedPath, nexusHome)
        } else if (rustBinary) {
            runRustAnalysis(rustBinary, geojsonPath, enrichedPath)
        } else {
            throw new RuntimeException(
                "Neither Python script nor Rust binary found in ${nexusHome}")
        }

        return enrichedPath
    }

    static Path findRustBinary(Path nexusHome) {
        def candidates = [
            nexusHome.resolve("target/release/nexus-forge-cyto-batch"),
            nexusHome.resolve("target/release/nexus-forge-cyto-batch.exe"),
            nexusHome.resolve("target/release/nexus-core-cli"),
            nexusHome.resolve("target/release/nexus-core-cli.exe"),
        ]
        return candidates.find { Files.isExecutable(it) || Files.exists(it) }
    }

    static void runPythonAnalysis(scriptPath, geojsonPath, enrichedPath, nexusHome) {
        def python = System.getenv("NEXUS_PYTHON") ?: "python"
        def cmd = [
            python, scriptPath.toString(),
            "--input", geojsonPath.toString(),
            "--output", enrichedPath.toString()
        ]

        def proc = new ProcessBuilder(cmd)
            .directory(nexusHome.toFile())
            .redirectErrorStream(true)
            .start()

        def exitCode = proc.waitFor(120, TimeUnit.SECONDS)
        if (!exitCode || proc.exitValue() != 0) {
            def stderr = proc.inputStream.text
            throw new RuntimeException("Python analysis failed (exit=${proc.exitValue()}): ${stderr}")
        }
        if (!Files.exists(enrichedPath) || Files.size(enrichedPath) == 0) {
            throw new RuntimeException("Enriched output is empty — analysis may have failed silently")
        }
    }

    static void runRustAnalysis(rustBinary, geojsonPath, enrichedPath) {
        // Create temp input dir structure for Rust batch CLI
        // Note: the Rust batch CLI expects a directory of JSON files, not GeoJSON.
        // For production use, pre-process via coord_adapter.py.
        throw new RuntimeException(
            "Direct Rust analysis not yet implemented. " +
            "Please install Python and nexus-forge-cyto package.")
    }

    // ── Load enriched results back into QuPath ──────────────────────────

    static void loadEnrichedResults(QuPathGUI qupath, imageData, Path enrichedPath) {
        def json = new groovy.json.JsonSlurper().parse(enrichedPath.toFile())

        def cells = json.cells
        def clinical = json.clinical
        def spatialFeatures = json.spatial_features

        if (!cells || !clinical) {
            throw new RuntimeException("Enriched JSON missing 'cells' or 'clinical' arrays")
        }

        def malignantClass = imageData.getServer()
            .getPathClass()
            ?.getPathClass("Nexus: Malignant")
        def normalClass = imageData.getServer()
            .getPathClass()
            ?.getPathClass("Nexus: Normal")

        def newAnnotations = []

        cells.eachWithIndex { ring, idx ->
            def status = clinical[idx]
            def color = status == "Malignant" ? Color.RED : new Color(30, 136, 229)
            def pathClass = status == "Malignant" ? malignantClass : normalClass

            def points = ring as List<List<Double>>
            def xCoords = points.collect { it[0] as double }
            def yCoords = points.collect { it[1] as double }

            def roi = ROIs.createPolygonROI(xCoords as double[], yCoords as double[], -1, 0, 0)
            def annotation = new PathAnnotationObject(roi, pathClass)
            annotation.setColorRGB(color.getRGB())

            // Add measurements
            if (spatialFeatures && idx < spatialFeatures.size()) {
                def sf = spatialFeatures[idx]
                def ml = annotation.getMeasurementList()
                ml.put("Nexus: Area", sf.area ?: 0.0)
                ml.put("Nexus: Circularity", sf.circularity ?: 0.0)
                ml.put("Nexus: Eccentricity", sf.eccentricity ?: 0.0)
                ml.put("Nexus: KNN Density", sf.knn_density ?: 0.0)
                ml.put("Nexus: RBF Risk Score", sf.rbf_risk_score ?: 0.0)
                ml.put("Nexus: Tumor Edge Dist", sf.distance_to_tumor_edge ?: 0.0)
                ml.put("Nexus: Perimeter", sf.perimeter ?: 0.0)
                ml.put("Nexus: Clinical", status)
                annotation.getMeasurementList().closeList()
            }

            newAnnotations.add(annotation)
        }

        // Remove old Nexus annotations and add new ones
        def hierarchy = imageData.getHierarchy()
        def toRemove = hierarchy.getAnnotationObjects().findAll {
            it.getPathClass()?.toString()?.startsWith("Nexus:")
        }
        hierarchy.removeObjects(toRemove, true)
        hierarchy.addObjects(newAnnotations)
        hierarchy.fireHierarchyUpdate()
    }
}


// ── Extension Registration ──────────────────────────────────────────────
//
// This class must be discoverable by QuPath's ServiceLoader.
// Ensure the file is in a directory that QuPath scans for extensions.
// Typically: ~/QuPath/extensions/nexus-forge/