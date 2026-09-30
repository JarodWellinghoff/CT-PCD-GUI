from pathlib import Path

import vtkmodules.all as vtk


class VtkMeshExporter:
    def export(
        self,
        polydata: vtk.vtkPolyData,
        path: Path,
    ) -> Path:
        suffix = path.suffix.lower()

        if suffix == ".ply":
            writer = vtk.vtkPLYWriter()
            writer.SetFileTypeToBinary()
        elif suffix == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetDataModeToBinary()
        elif suffix == ".stl":
            writer = vtk.vtkSTLWriter()
            writer.SetFileTypeToBinary()
        else:
            raise ValueError(f"Unsupported mesh format: {suffix}")

        writer.SetFileName(str(path))
        writer.SetInputData(polydata)

        if writer.Write() != 1:
            raise OSError("VTK did not confirm that the mesh was written.")

        return path
