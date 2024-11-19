# crocoddyl-multithreading-benchmark
Benchmark intended to trace issues related to slowdown of single-treaded sections of Crocoddyl code when multithreading is enabled.

## Set up

The project is created to be used with [dev-container](https://code.visualstudio.com/docs/devcontainers/containers) in Visual Studio Code.

Clone the repository:
```bash
git clone https://github.com/Kotochleb/crocoddyl-multithreading-benchmark.git --recursive
```

Install [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) plugin in VSCode and click **Open in container**. The build will start automatically and Docker image with all required dependencies will be set up.

> [!WARNING]
> All dependencies including [Crocoddyl](https://github.com/loco-3d/crocoddyl) and [Pinocchio](https://github.com/stack-of-tasks/pinocchio) are built from source with maximum optimization. This means the image can take up to 30 minutes to build and will require 32 Gb of RAM to do so. To reduce memory footprint during build modify [Dockerfile](./.devcontainer/Dockerfile) and change last value in `install_dependency.sh`.

After the docker image is build inside of container run
```bash
./scripts/build.sh <clang++-15 | g++ | icpx>
```
to build/rebuild the project.

Binary outputs of the scripts can be found in the `build/experiments/<experiment name>` folders.

## Repository structure

This repository is split into series of experiments. Each experiment is in it's separate folder with subfolders contaning source code of the experiment and Jupyter notebooks used to postprocess data and document results.

## Tools installed

The docker image provides:
- [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [Valgrind](https://valgrind.org/)
- [KCachegrind](https://kcachegrind.sourceforge.net/html/Home.html)
- [Massif-Visualizer](https://github.com/KDE/massif-visualizer)
- [LTTng](https://lttng.org/)
- [Trace Compass](https://eclipse.dev/tracecompass/)
