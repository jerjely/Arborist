# Arborist

*Formerly SC_morph.* Morphology of superior colliculus neurons in the laminar frame: depth from the brain
surface, position relative to the CR border, branching, Sholl, AIS and classification.

**Coming from SC_morph:** unzip Arborist into its own folder and run `install.bat` (Mac: `install.command`)
once. It reuses the existing Python environment (no long download), replaces the old shortcut, and takes
over the data folder setting and any files still waiting to be copied to the USB stick. Your data is
untouched: projects (`…_SC_morph.json`), results (`SC_morph_results`) and the database file keep their
names, so every cell opens as before. The old SC_morph folder can be deleted afterwards.

**Install:** use the installers on the Releases page (below); `install.bat` / `install.command`
run Arborist from this source folder instead.

Desktop tool (napari-based; tracing stays in Fiji/SNT) for measuring where a filled SC neuron sits relative to the brain
surface and to the lower edge of the Calretinin-positive (sensory) SC. It replaces the
*surface_to_swc_distance* notebook and the Fiji point-tool step.

## Install (recommended): download the installer
Get the installer for your computer from the repository's **Releases** page (right-hand side on GitHub):

| Computer | File |
|---|---|
| Windows 10/11 | `Arborist-<version>-Windows-x86_64.exe` |
| Mac with Apple silicon (M1/M2/M3/M4) | `Arborist-<version>-MacOSX-arm64.pkg` |
| Intel Mac | `Arborist-<version>-MacOSX-x86_64.pkg` |

Each installer contains its own Python, napari and everything else (no Anaconda, no internet needed during
install, no administrator rights). It is a large download (~600 MB) and takes 1-3 minutes to install.

**Windows:** run the .exe. Windows SmartScreen may say *"Windows protected your PC"* because the installer
isn't code-signed: click *More info* > *Run anyway*. Arborist goes to `%LOCALAPPDATA%\Programs\Arborist`, with a
shortcut on the Desktop and in the Start menu. Uninstall from *Settings > Apps*.

**macOS:** double-click the .pkg. Because it isn't signed by an Apple developer account, macOS may refuse
the first time: right-click the .pkg > *Open* > *Open* (on macOS 15+: *System Settings > Privacy & Security* >
*Open Anyway*). Choose *Install for me only*. Arborist.app appears in `~/Applications` and Launchpad.
To uninstall, delete `~/Applications/Arborist.app` and `~/Library/Arborist`.

Messages and errors go to `~/Library/Logs/Arborist.log` (Mac) or `%LOCALAPPDATA%\Arborist\Arborist.log` (Windows).
Updating: install the new version over the old one. Settings, the data folder choice and your data are kept.

### Making a new release (for the maintainer)
1. Raise `__version__` in `arborist/__init__.py` (e.g. `"1.0.1"`) and commit.
2. `git tag v1.0.1 && git push && git push --tags`
3. GitHub Actions (*Actions* tab, "Build installers") builds the three installers on GitHub's Windows and Mac
   machines in ~15 minutes and attaches them to a new Release. *Run workflow* on that page builds them
   without a release (download from the run's *Artifacts*).

How it works: `packaging/recipe` turns Arborist into a conda package (rattler-build) with its Start-menu /
Applications shortcut (`packaging/recipe/menu/arborist.json`); `packaging/construct.yaml` bundles it with
Python and all dependencies from conda-forge into the .exe/.pkg (conda *constructor*).

## Install from source (Windows)
1. Double-click `install.bat` once. If Miniconda/Anaconda is installed it creates a conda
   environment called `Arborist` (or reuses `SC_morph` from before the rename); otherwise it uses (or installs) plain Python 3.12 and makes a
   `.venv` folder here. The first install downloads napari etc. and takes a few minutes.
2. Start the app from the **Arborist** shortcut that the installer puts on the Desktop and in the
   Start menu (or with `run_Arborist.bat`). If the shortcut is missing, run `make_shortcut.bat`.

## Install from source (macOS)
1. Unzip anywhere (Downloads is fine) and double-click `install.command`. The first time macOS
   blocks files from the internet: right-click it > *Open* > *Open*.
2. The installer copies the program to `~/Library/Application Support/Arborist` (apps started from
   Finder may not read Downloads/Documents/Desktop) and uses Miniconda/Anaconda/Miniforge if installed
   (conda environment `Arborist`), otherwise Python 3.10+ (environment inside that folder).
3. It creates **Arborist.app** (with the logo) in `~/Applications` (also in Launchpad) plus a link on
   the Desktop; drag it to the Dock if you like. The downloaded folder can be deleted afterwards.
4. To update: download the new version and run its `install.command` again (the environment is reused,
   so this is quick).

The first time you open a stack on the external drive macOS asks whether Arborist may access
removable volumes - allow it. Messages and errors go to `~/Library/Logs/Arborist.log`;
`Arborist.command` starts the installed program in a Terminal window instead.
Zoom with **⌘+scroll** or **Control+scroll**; two-finger scroll moves through the stack.

## Running by hand
From an Anaconda Prompt / Terminal: `conda activate Arborist`, `cd` to this folder,
`python -m arborist`.

## Database (your data folder)
Open the **Database** tab and choose the folder that holds the experiment-day folders (e.g.
`D:\SC cells`). *⇱ Big window* shows it in its own window. Expected layout:

```
<day>\<Section N>\whole section\   10x whole-section scan   (else: 10x file in the Section folder)
                 \SC\              20x of the whole SC, both hemispheres (or a 20x file directly in the
                                   Section folder, e.g. ..._SC_20x.czi)
                 \Left | Right\    63x original .czi, 63x ..._Stitch.czi, ...ankyrin....czi,
                                   <Stitch name>.swc, <Stitch name>_recon.tif, extra fills (*.tif)
```

- The cell is the `Stitch` .czi (else the original 63x). In Left/Right only .czi files count as scans;
  `_recon.tif` marks a finished reconstruction and other .tifs (axon.tif, dendrites.tif, full.tif...)
  are ignored as extra fills. A folder without any .czi is read from its TIFFs instead.
- 20x CR: the `SC` folder or a 20x scan directly in the Section folder (both count for left and
  right), otherwise a 20x .czi in Left/Right, otherwise missing.
- The ankyrin scan is any file with "ankyrin"/"AnkG" in its name.
- The .swc must start with the Stitch file's name (SNT's default); with several, the shortest
  name wins (`..._Stitch.swc` over `..._Stitch2.swc`).
- Magnification comes from the .czi objective metadata (TIFF: `10x`/`20x`/`63x` in the name, else
  pixel size).

Columns: 10x, 20x CR, 63x cell, AnkG scan, SWC, Recon, AIS measured, Surface, CR border, Done
(analysed); hover a cell to see the files. Recon is ticked by a `_recon.tif` or by the fill
projections `soma.tif` / `dendrites.tif` / `axon.tif`; hovering the 63x cell, SWC or Recon column
shows a colour preview of them (soma grey, dendrites red, axon green), cached as
`SC_morph_results\<cell>_fill_preview.png`. Double-click a row to open the cell: Stitch + 20x CR +
AnkG scan + .swc (or its saved project). Projects are saved next to the Stitch as
`<cell>_SC_morph.json` (automatically after *Run analysis*), results go to `SC_morph_results\`.
Notes are editable and stored in `SC_morph_database.json` in the root folder; *Export table* writes
CSV or Excel. Stacks above 1.5 GB (the stitched .czi files) are read slice by slice, so they open
quickly and do not need to fit in memory.

## Viewing
The **View** buttons at the top of the Arborist panel switch between the whole section (10x,
loaded on first click), the 20x CR stack, the 63x cell and the ankyrin-G scan; the layer list on the
left only shows the layers of the image on screen. Each tab also switches to the image it works on.
**Mouse wheel** moves through the z-stack, **Ctrl+wheel** (Mac: ⌘ or Control + scroll) zooms, drag pans. The **Display** box
sets brightness/contrast (min/max, gamma, *Auto* from the current slice) per channel.

## Workflow
| Tab | What you do |
|---|---|
| **1 Stacks** | Open the **cell stack** (high-mag, .czi or .tif) and any **low-mag stacks**. Voxel size is read from the file; check it, and set it by hand if a TIFF has no calibration. |
| **2 Surface** | On the 63x cell stack press **✎ Mark surface** (turns orange) and mark the pial surface on every 5th–10th slice: with **✍ Pen** on, drag along it (points every 5 µm; hold **Space** and drag to move the image); with the pen off, click points. *↶ Undo last line / point* removes the last stroke, *Clear this slice* the marks on the slice shown, *🗑 Clear surface* all surface marks plus the built surface (and its saved files). The surface is built slice by slice (follows any curvature, e.g. the whole SC dome), rebuilt automatically while you mark and saved next to the cell (`<cell>_SC_morph.json`, `<cell>_surface_points.csv`, `<cell>_brain_surface.stl`). The cyan line shows where it cuts the current slice. A Fiji `SURFACE.txt` in the Left/Right folder is read automatically. |
| **3 CR** | Opens the 20x low mag. Mark its surface and the lower edge of the CR band there (pen or clicks), then *Measure*. If the CR edge is visible on the 63x, switch *Mark on* to *High mag* and mark the CR points there - they are measured against the 63x surface from tab 2. Every measurement is made inside one image only (20x and 63x never share coordinates) and listed with the image it came from; tick which to use (mean) or type a depth. *Clear this slice*, *🗑 Clear surface* and *🗑 Clear CR points* delete marks of the image shown. The 63x then shows the CR border as a surface parallel to its brain surface at that depth (orange). |
| **5 AIS** | On the ankyrin-G scan (set the ankyrin and the cell-fill channel): (1) click the AIS start (soma side) and end - the AIS path follows the ankyrin staining between clicks and its length is the AIS length; (2) click where the axon-bearing process leaves the soma; (3) choose soma or dendrite origin; (4) for a dendrite, click the branch point. Distances from the soma exit point follow the fill signal (soma exit → branch point → AIS start; add optional bend points if it takes a wrong turn); straight-line values are reported too. Marks are saved automatically into the cell's project. |
| **4 Neuron** | Trace the cell in Fiji/SNT and *Load SWC* (or drag the .swc onto the window). It is drawn over the stack so you can check the overlay. **AIS (ankyrin G):** on the separate ankyrin scan (AnkG stack) mark the AIS from start to end, the soma centre and, for an axon-carrying dendrite, the branch point, and choose the origin; lengths and distances are measured in that scan (straight-line). Alternatives: trace the AIS in SNT as path type *Custom* (SWC type 7), or mark it on the cell stack itself (snapped onto the traced axon, origin found automatically). |
| **6 Analyze** | Optionally give the cell a *manual type label*. Writes everything to the output folder and opens the **Results** panel: rotatable **3D view** (brain surface, CR border surface, SWC paths), **Laminar %** (dendrite/axon length per 10 % depth band, 0 % = brain surface, 100 % = CR border), side view, **Sholl profile** (intersections, branch points + tips, and dendrite length per shell), **Sholl 3D** (rotatable shells with every crossing marked), **Sholl × depth** (where crossings sit between surface and CR border), depth in µm, and a readable summary. A **Plot options** box above the tabs changes laminar bin width, % vs µm, dendrites/axon on/off, a smooth line, Sholl shell size, depth bins and sphere count; *Save these figures…* writes PNG + SVG tagged with the settings. |
| **7 Classify** | Loads `all_cells_summary.csv` (one row per analysed cell), lets you pick the features, and clusters the cells (Ward on z-scored features): dendrogram, PCA plot, feature heatmap, `_clusters.csv`. Manual labels are shown next to the clusters. |

*Save project* stores all points, settings and the loaded neuron in a `.json` file, so you can continue later.

## Outputs (per cell, prefix = cell name)
- `_summary.csv`: soma depth, CR-border depth, dendrite/axon length, length-weighted depth
  percentiles, length and fraction inside the sensory (CR+) zone. Each run also adds or updates a
  row in `all_cells_summary.csv`.
- Morphology features in the laminar frame (also in the summary row): soma depth (absolute and
  normalized to the CR border); dendritic field width parallel to the surface (major/minor in-plane
  axis); height perpendicular to it; vertical aspect; dorsal/ventral reach; fraction of dendrite
  above the soma; dendrite centroid offset; hull area/volume; number of primary dendrites, branch
  points and tips; Sholl peak; axon length and direction.
- **Dendrite thickness** (mean, primary, tip diameter and taper ratio) comes from the SWC radius column. It is only meaningful when radii were fitted in SNT (*Refine/Fit paths*); if every radius is the same the features are left empty and `dend_radii_fitted` is False.
- AIS: length (along the axon and straight), start/end distance from the soma (along the neurite
  and straight), origin (`soma` or `dendrite`, i.e. axon-carrying dendrite), for dendrite origin
  how far from the soma the axon branches off, AIS depth and angle to the surface normal.
- Laminar fractions of dendrite / axon length: above surface, 0–25 %, 25–50 %, 50–75 %, 75–100 %,
  below the CR border (`dend_frac_*`, `axon_frac_*`) and the full 10 % profile
  (`_laminar_distribution.csv`).
- Branching: branches (sections between branch points/tips), max branch order, mean branch and
  terminal branch length, tortuosity (path length / straight distance soma → tip), mean bifurcation
  angle; soma diameter from the SWC soma contour or radius (`soma_size_source` says which - SNT
  writes a default radius if the soma was not sized).
- Clustering uses by default: soma normalized depth, field width/height/aspect, dorsal fraction and
  reach, planar elongation, primary dendrites, branch points, branches, max branch order, length,
  tortuosity, Sholl peak radius and the four laminar fractions (tick/untick in tab 7).
- `type_hint`: rule-of-thumb wide-field / narrow-field / horizontal / stellate / marginal call with
  the reason. The thresholds (`HINT_RULES` in `arborist/morphometry.py`) are starting guesses; tune
  them on your labelled cells. Treat the hint as a suggestion, not a classification.
- `_nodes.csv`: per SWC node: depth (Method 1, mesh), plane depth (Method 2), tilt-corrected z
  (Method 3), normalized depth (0 = surface, 1 = CR border), zone.
- `_segments.csv`: per edge length + midpoint depth (used for length-weighted stats).
- `_sholl.csv`, figures as PNG + SVG (laminar side view, depth histograms, Sholl).
- `_brain_surface.stl`, `_CR_border_surface.stl`, `_neuron_colored.ply`, interactive `_3d.html`.


**Building the surface:** draw the surface line on as many slices as you like; each line is saved
straight away, but the surface is only rebuilt when you press **Build surface** (it turns orange and shows
how many slices have lines when there is something new to build). *Rebuild after every line* brings back
the automatic rebuild. Analyze builds the surface first if it is out of date. Use the *Follow slices*
method for curved surfaces; Delaunay (from the old notebook) folds over on a dome.

### Working from a USB stick

Writing to a USB stick is slow, so by default Arborist saves the project, surface files, results,
previews and the database **on this computer while you work**, and copies them to the stick when you
close the app (a progress bar shows the copy). The *Saving* box in tab 1 shows how many files are waiting
and has a **Copy now** button. If the stick is unplugged when you close, Arborist says so and keeps the
files safe on this computer; they are copied the next time it starts. The local copies live in
`%LOCALAPPDATA%\Arborist\work` (Windows) or `~/Library/Caches/Arborist/work` (Mac).
Options: only USB / external drives (default), always, or save straight into the folders.
The images themselves are still read from the stick.

### Automatic first draft of a tracing (tab 4)

Press **① Click the soma…**, scroll to the slice where the soma is brightest and click it. Arborist reads the
fill channel in a box around the soma, removes the background, keeps what is brighter than *Sensitivity* ×
noise and connected to the soma, and follows it outwards, longest branches first. The draft is drawn over
the image and saved as `<cell>_autotrace.swc`.

It is a **draft**: all branches are tagged dendrite, touching neighbours may be joined on and dim branches
may be missing. Open it in SNT, correct it, tag the axon and save it under the cell's normal `.swc` name.
Drafts do not count as "SWC done" in the Database and are not stored in the project.
*Trace again with these settings* re-runs with another sensitivity / branch length without re-reading the image.

### Batch re-analysis, diameters from the image, and comparing groupings

- **Re-analyse all cells…** (tab 7) runs Analyze again for every cell that has a surface and a tracing,
  from its saved project and .swc - the images are not opened. AIS values measured on the ankyrin scan are
  kept from the previous results. The pooled table is rebuilt afterwards.
- **Measure diameters** (tab 4) measures the width of the filled neurite at every traced point (width at
  half brightness across the neurite, in the optical section). The values are saved next to the cell as
  `<cell>_radii_from_image.csv`, put back on the tracing whenever the cell is opened, and used for the
  diameter features. The width includes the optical blur, so the thinnest branches read slightly thick.
- **Excel copies:** the pooled table and the cluster tables are also written as `.xlsx` (a CSV opens as one
  column in Excel with decimal commas). The pooled table starts with `cell, day, section, side`.
- **In vivo light responses:** a second table can be linked next to the electrophysiology sheet (one row
  per cell recorded in vivo and reconstructed). Numeric columns become `invivo_…` features; a cluster or
  type column becomes a label.
- **Stability tab:** the clustering is repeated with each cell left out in turn; the map shows how often two
  cells end up together, and the summary says what share of cell pairs keep their relation.
- **vs labels tab:** clusters against the label chosen under *Label cells by* (your types, ephys types,
  in vivo clusters), with the adjusted Rand index (1 = same grouping, 0 = chance).
- **PCA tab:** arrows show the features that separate the cells most; names are shortened to the part
  that differs between cells.

### Safety and speed notes (November review)

- **Projects:** each cell saves into its own project file; a project is written to a temporary file and
  swapped in, so a crash while saving cannot damage it. A project that cannot be read is kept as
  `…_SC_morph.json.unreadable` and the cell opens fresh.
- **USB copy:** a local copy never overwrites a newer file on the drive; if the stick gets another drive
  letter (or the top folder was renamed) the waiting files find their folders again; *Show…* opens the
  folder with the files still waiting.
- **Autosave** writes only what changed and no longer re-reads the whole database after every mark; the
  Database table refreshes a few seconds later, or when it is next shown.
- **SWC files** are checked when loaded: unreadable lines, duplicate ids, untagged paths, no soma, or a
  tracing that does not sit inside the image (pixel units / another voxel size) are reported.
- `python tests/run_all.py` runs every test with timings.

### Large stitched files

Stacks over ~1.5 GB are read one plane at a time. Recently viewed planes stay in memory (up to a quarter
of the RAM, max 4 GB), and the next plane in the direction you scroll is read in the background, so
stepping back and forth while marking does not re-read the file. Planes wider than 4096 px are shown as an
image pyramid: zoomed out you see a reduced copy, zoomed in only the visible part at full resolution.
The CR line on a new slice is drawn once scrolling pauses.

**Pooling cells for classification:** each cell's results stay next to that cell. *Collect all analysed
cells* (tab 7) gathers every analysed cell under the database folder into one table,
`SC_morph_results/all_cells_summary.csv` in the root folder, keeping the newest analysis of each cell.
The table is rebuilt each time it is loaded, so newly analysed cells are always included; *Run clustering*
also collects automatically when the chosen table has fewer than 3 cells.

### Slice electrophysiology in the classification

Tab 7 → **Link electrophysiology table…** takes an Excel (.xlsx/.xls, any sheet) or CSV file with one row
per cell. The sheet can be written however is convenient:
- title rows above the header and a units row under it (`mV`, `MΩ`) are recognised; units go into the
  column name;
- decimal commas (`-62,5`) are read as numbers;
- the cell is identified by any mix of **date**, **section**, **side** (L/R, Left/Right, B/J, bal/jobb)
  and **cell number**, or by a **file name**. Dates can be Excel dates, `2026.03.12.`, `20260312` and so on.

Rows are matched to analysed cells through each cell's image path (day folder / Section N / Left|Right /
file name), which every new analysis stores in the `source_path` column. A row that fits exactly one cell
is matched automatically; anything else can be set with the drop-down in the dialog, and those choices
are kept in `all_cells_summary_ephys_link.json`. Every other numeric column becomes an `ephys_…` feature.

When clustering, the **Morphology / Ephys / Both / None** buttons tick feature sets. *Weigh morphology
and ephys equally* scales each group so the one with more columns does not dominate. *Leave out cells
without ephys data* skips cells with no matched row, instead of filling their values with the median.
The merged table is saved as `…_clusters_table_with_ephys.csv`.

## Figure style
All figures follow the lab's SC poster (Helsinki Vision 2026): Inter (DejaVu Sans if Inter is not
installed), dark-slate text, pale dome fill for the sensory SC, dashed grey line for the CR border,
dendrites red (#c0392b), axon green (#1a9c8a - shifted slightly towards teal so it stays distinct from
the red for colour-blind readers), soma grey, the poster's cell-type palette for clusters. SVGs keep
editable text. The colours live in `arborist/style.py`.

## Conventions
- Coordinates are µm, same as SNT `.swc`: first slice is z = 0.
- Positive depth means into the brain. The brain side is detected automatically: towards the soma
  if one exists, otherwise towards the CR border points, otherwise towards the brighter tissue.
  If depths come out negative, use *Flip brain side*.
- Fiji point tables: `Slice` is converted as (Slice − 1) × z-step, so the points line up with SNT
  `.swc` coordinates. (The old notebook used Slice × z-step, which put the surface one z-step deeper.)
- The default surface follows the marked line on each slice and joins neighbouring slices. Depth is
  measured **in the section plane** (distance to the surface line on the same optical section), and the
  CR border is drawn as a parallel line on every slice (offset sideways, never along z) - so thin 63x
  stacks with a deep CR border stay consistent. It works for strongly curved surfaces (on a synthetic dome turning 140° the depth error is < 2 µm;
  the old height-field / Delaunay surfaces were off by > 100 µm on the steep part). During analysis
  the lines are extended past every neurite. *Height field* and *Delaunay (raw points, old notebook)*
  are still available for comparison.
