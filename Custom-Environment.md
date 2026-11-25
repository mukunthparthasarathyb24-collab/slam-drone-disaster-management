# Custom Unreal Environment (IIITS Digital Twin)

The custom Unreal Engine simulation environment for the IIIT Sri City campus is hosted on GitLab instead of this repository due to the size of the Unreal project assets.

## Repository

- **GitLab Link:** https://gitlab.com/Adamkoda2306/IIITS-Digital-Twin

## Getting the Environment

1. Clone the repository from GitLab:

   ```bash
   git clone https://gitlab.com/Adamkoda2306/IIITS-Digital-Twin.git
   ```

2. If the repository uses Git LFS for large Unreal assets (`.uasset`, `.umap`, etc.), make sure LFS objects are pulled:

   ```bash
   cd IIITS-Digital-Twin
   git lfs pull
   ```

3. Open the Unreal Engine project file (`.uproject`) in Unreal Engine, build/launch the editor if prompted, and press **Play** (with AirSim configured as the game mode / multirotor vehicle) before running any simulation, data-collection, or navigation scripts from this repository.

## Notes

- Keep AirSim's `settings.json` configured on your machine (typically in `~/Documents/AirSim/settings.json` on Windows/macOS/Linux depending on your setup) so the multirotor vehicle and sensors (camera, LiDAR) match what your scripts expect.
- Do not commit large Unreal assets, packaged builds, or recorded simulation data into this GitHub repository.
