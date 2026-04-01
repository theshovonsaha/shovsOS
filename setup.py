import os
import shutil

from setuptools import Command, find_namespace_packages, setup


class _FallbackBdistWheel(Command):
    description = "Fallback bdist_wheel command for offline metadata generation"
    user_options: list[tuple[str, str, str]] = []

    def initialize_options(self) -> None:
        return None

    def finalize_options(self) -> None:
        return None

    def run(self) -> None:
        raise RuntimeError("Building wheels requires the 'wheel' package to be installed.")

    @staticmethod
    def egg2dist(egg_info: str, dist_info_dir: str) -> None:
        if os.path.isdir(dist_info_dir):
            shutil.rmtree(dist_info_dir)
        shutil.copytree(egg_info, dist_info_dir)
        pkg_info = os.path.join(dist_info_dir, "PKG-INFO")
        metadata = os.path.join(dist_info_dir, "METADATA")
        if os.path.exists(pkg_info):
            shutil.copyfile(pkg_info, metadata)
        wheel_file = os.path.join(dist_info_dir, "WHEEL")
        if not os.path.exists(wheel_file):
            with open(wheel_file, "w", encoding="utf-8") as handle:
                handle.write("Wheel-Version: 1.0\n")
                handle.write("Generator: setuptools-fallback\n")
                handle.write("Root-Is-Purelib: true\n")
                handle.write("Tag: py3-none-any\n")


PACKAGE_PATTERNS = [
    "api*",
    "config*",
    "engine*",
    "guardrails*",
    "llm*",
    "memory*",
    "orchestration*",
    "plugins*",
    "services*",
    "shovs_memory*",
]


setup(
    name="shovs-memory",
    version="0.1.0",
    description="Inspectable agent memory with deterministic user facts, temporal invalidation, and semantic retrieval.",
    long_description=(
        "Shovs memory layer with deterministic fact extraction, temporal invalidation, "
        "semantic retrieval, and inspectable memory state."
    ),
    long_description_content_type="text/plain",
    author="Shovon Saha",
    python_requires=">=3.10",
    install_requires=[
        "httpx",
        "numpy",
        "python-dotenv",
    ],
    include_package_data=True,
    packages=find_namespace_packages(where=".", include=PACKAGE_PATTERNS),
    cmdclass={"bdist_wheel": _FallbackBdistWheel},
)
