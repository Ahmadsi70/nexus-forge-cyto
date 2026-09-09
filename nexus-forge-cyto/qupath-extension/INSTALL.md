# QuPath Extension — Step-by-Step Installation

## Prerequisites

Before installing the Nexus-Forge QuPath Extension, ensure you have:

1. **QuPath** >= 0.5.0 installed ([download](https://qupath.github.io/))
2. **Java 17+** (bundled with QuPath 0.5+)
3. **Python** >= 3.10 with `nexus-forge-cyto` dependencies installed
4. A working clone of `nexus-forge-cyto` (this repository)

## Installation Steps

### Step 1: Locate Your QuPath Extensions Directory

QuPath looks for extensions in specific locations depending on your OS:

| OS | Extensions Path |
|----|----------------|
| **Windows** | `C:\Users\<YourName>\QuPath\extensions\` |
| **macOS** | `~/QuPath/extensions/` |
| **Linux** | `~/QuPath/extensions/` |

If the `extensions/` directory does not exist, create it.

### Step 2: Copy the Extension

Copy the entire `qupath-extension/` directory from this repository into your
QuPath extensions folder:

```bash
# Example (Linux/macOS)
cp -r nexus-forge-cyto/qupath-extension/ ~/QuPath/extensions/nexus-forge/

# Example (Windows PowerShell)
Copy-Item -Recurse nexus-forge-cyto\qupath-extension\ $env:USERPROFILE\QuPath\extensions\nexus-forge\
```

The directory structure should look like:

```
~/QuPath/extensions/nexus-forge/
  ├── NexusForgeExtension.groovy
  ├── README.md
  └── INSTALL.md
```

### Step 3: Set the NEXUS_FORGE_HOME Environment Variable

This tells the extension where to find the Nexus-Forge pipeline code.

**Windows (PowerShell):**
```powershell
[System.Environment]::SetEnvironmentVariable('NEXUS_FORGE_HOME', 'C:\Users\YourName\cancer_project', 'User')
```

**Linux/macOS (add to ~/.bashrc or ~/.zshrc):**
```bash
export NEXUS_FORGE_HOME="$HOME/cancer_project"
```

### Step 4: Verify Python Dependencies

```bash
cd $NEXUS_FORGE_HOME/nexus-forge-cyto
pip install -r requirements-api.txt
pip install -r requirements-dashboard.txt
```

### Step 5: Restart QuPath

Close and reopen QuPath. After restart, you should see a new menu item:

**Extensions → Nexus-Forge → Analyze with Nexus-Forge**

## Verifying Installation

1. Open any slide image in QuPath
2. Run built-in cell detection: **Analyze → Cell detection → Cell detection**
3. Click **Extensions → Nexus-Forge → Analyze with Nexus-Forge**
4. After processing, you should see color-coded annotations (red/blue) with
   measurement values in the Measurements tab.

## Common Issues

### "Nexus-Forge not found" error

The extension cannot locate your `cancer_project` directory. Verify that
`NEXUS_FORGE_HOME` points to a directory containing `Cargo.toml`.

### Python not found

If Python is not on PATH, set the `NEXUS_PYTHON` environment variable to the
full path of your Python 3.10+ executable.

### No annotations appear after analysis

Check that your cell annotations have >= 3 vertices (points). Single-point or
two-point annotations are skipped. Also check that you're not zoomed out past
the annotation visibility threshold.

### Out of memory

Large slides with thousands of detected cells may require increasing QuPath's
memory limit. Edit `QuPath.cfg` or pass `-Xmx8g` to the Java VM.