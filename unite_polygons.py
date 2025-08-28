import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import Literal, Union
import os, argparse

import logging as log
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = log.getLogger(__name__)

_OP_TYPE = Literal[Polygon.union, Polygon.intersection]
class polygon_type:
    OUTER = "outer" # polygon_type.OUTER if Clockwise
    HOLE = "hole" # polygon_type.HOLE if Counter-Clockwise

class simple_polygon:
    def __init__(self, parray:list=[], apply_simplifaction=False):
        self.vertices = parray
        self._signed_area = None
        self.orientation = None
        self.perimeter = None
        self.get_area()
        self.get_perimeter()
        
        if apply_simplifaction:
            self.polygon_simplification()
    
    def get_perimeter(self) -> float:
        """
        Calculates and returns the perimeter of the polygon.
        The perimeter is computed as the sum of the Euclidean distances between consecutive vertices.
        The result is cached for subsequent calls.
        Returns:
            float: The perimeter of the polygon.
        """
        if self.perimeter is not None:
            return self.perimeter
        
        perimeter = 0.0
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]
            perimeter += np.linalg.norm(p2 - p1)
        self.perimeter = perimeter
        return perimeter
    
    def calculate_signed_area(self) -> float:
        """
        Calculates the signed area of the polygon using the Shoelace formula.

        The signed area is positive if the vertices are ordered counterclockwise (inner polygon),
        and negative if they are ordered clockwise (outer polygon).

        Returns:
            float: The signed area of the polygon.
        """
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        # Shoelace formula (signed area)
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        area += 0.5 * (x[-1] * y[0] - x[0] * y[-1])  # closing term
        self._signed_area = area
        return area

    def get_area(self, absolute=True) -> float:
        """
        Returns the area of the polygon.

        Args:
            absolute (bool, optional): If True, returns the absolute value of the area.
                If False, returns the signed area. Defaults to True.

        Returns:
            float: The area of the polygon. If `absolute` is True, returns the absolute area;
                otherwise, returns the signed area.

        Notes:
            The signed area is positive if the vertices are ordered counterclockwise (inner polygon),
            and negative if ordered clockwise (outer polygon).
        """
        if self._signed_area is None:
            self.calculate_signed_area()
        return abs(self._signed_area) if absolute else self._signed_area

    def calculate_orientation(self) -> polygon_type | None:
        """
        Determines the orientation of the polygon based on its signed area.
        Returns:
            polygon_type.HOLE: If the signed area is positive, indicating a hole.
            polygon_type.OUTER: If the signed area is negative, indicating an outer boundary.
            None: If the signed area is zero, indicating a degenerate polygon.
        """
        if self._signed_area is None:
            self._signed_area = self.calculate_signed_area()
        s_area =self.get_area(absolute=False)
        return polygon_type.HOLE if s_area > 0 else polygon_type.OUTER if s_area < 0 else None
    
    def get_orientation(self) -> polygon_type | None:
        """
        Returns the orientation of the polygon.
        If the orientation has not been calculated yet, it computes and caches the orientation
        using the `calculate_orientation` method. Subsequent calls return the cached value.
        Returns:
            polygon_type | None: The orientation of the polygon, or None if it cannot be determined.
        """

        if self.orientation is None:
            self.orientation = self.calculate_orientation()
        return self.orientation
    
    def rotate(self, angle_deg: float, origin=(0, 0)):
        """
        Rotates the polygon by a specified angle around a given origin.
        Parameters:
            angle_deg (float): The rotation angle in degrees.
            origin (tuple, optional): The (x, y) coordinates of the rotation origin. Defaults to (0, 0).
        Modifies:
            self.vertices: Updates the polygon's vertices to their new rotated positions.
        Example:
            polygon.rotate(90, origin=(1, 2))
        """

        angle_rad = np.deg2rad(angle_deg)
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        
        shifted = self.vertices - origin
        rotated = shifted @ rot_matrix.T
        self.vertices = rotated + origin

    def simplify_collinear(self, tol=1e-6):
        """
        Simplifies the polygon by removing collinear vertices within a specified tolerance.
        This method iterates through the polygon's vertices and removes any vertex that is collinear
        with its adjacent vertices, as determined by the area of the triangle formed by the three points.
        Vertices are retained only if the area exceeds the given tolerance.
        Args:
            tol (float, optional): Tolerance for determining collinearity. Vertices are considered
                collinear if the area of the triangle formed by three consecutive points is less than
                or equal to this value. Defaults to 1e-6.
        Modifies:
            self.vertices: Updates the list of vertices to exclude collinear points, provided the
                resulting polygon has more than three vertices.
        """
        if len(self.vertices) < 3:
            return
        
        simplified = []
        n = len(self.vertices)
        for i in range(n):
            p_prev = self.vertices[i-1]
            p_curr = self.vertices[i]
            p_next = self.vertices[(i+1) % n]

            # Vector cross product (area of triangle)
            area = abs(np.cross(p_next - p_curr, p_prev - p_curr)) / 2.0
            if area > tol:  # keep only if not collinear
                simplified.append(p_curr)
        
        if len(simplified) >= 3:
            self.vertices = simplified
    
    def simplify_by_distance(self, threshold: float = 1e-2):
        """
        Simplifies the polygon by removing consecutive vertices that are closer than a given threshold.
        Args:
            threshold (float, optional): The minimum distance required between consecutive vertices.
                Vertices closer than this value are removed. Defaults to 1e-2.
        Notes:
            - The polygon is assumed to be represented by self.vertices, a list or array of coordinates.
            - The method ensures the polygon remains closed by appending the first vertex at the end if necessary.
            - If the simplified polygon has fewer than 4 vertices (including closure), the original vertices are retained.
        """

        if len(self.vertices) < 3:
            return  # not enough to simplify
        
        simplified = [self.vertices[0]]
        for i in range(1, len(self.vertices)):
            if np.linalg.norm(self.vertices[i] - simplified[-1]) >= threshold:
                simplified.append(self.vertices[i])

        # Ensure closed polygon (last == first)
        if not np.allclose(simplified[0], simplified[-1]):
            simplified.append(simplified[0])

        if len(simplified) >= 3:
            self.vertices = simplified
    
    def apply_polygon_simplification(self, collinear_tol=1e-6, distance_threshold=1e-2):
        """
        Simplifies the polygon by reducing unnecessary vertices.
        This method applies two simplification steps:
        1. Collinear simplification: Removes vertices that are collinear within a specified tolerance.
        2. Distance simplification: Removes vertices that are closer than a specified distance threshold.
        Args:
            collinear_tol (float, optional): Tolerance for detecting collinear points. Defaults to 1e-6.
            distance_threshold (float, optional): Minimum distance between vertices to retain them. Defaults to 1e-2.
        Logs:
            Information and debug messages about the simplification process and the number of vertices before and after each step.
        """
        
        log.info(f"Applying simplification to polygon with {len(self.vertices)} vertices.")
        self.simplify_collinear(tol=collinear_tol)
        log.debug(f"Vertices after collinear simplification: {len(self.vertices)}")
        self.simplify_by_distance(threshold=distance_threshold)
        log.debug(f"Vertices after distance simplification: {len(self.vertices)}")
        log.info(f"Simplification complete. Polygon now has {len(self.vertices)} vertices.")

class complex_polygon:
    def __init__(self, polygon_list:list=[]):
        self.area = None
        self.perimeter = None   
        self.polygons = []
        if polygon_list is None:
            raise ValueError("polygon_list must contain at least one polygon")
        self.polygons = [simple_polygon(p) for p in polygon_list]
        self.area = self.get_area()
        self.polygons.sort(key=lambda poly: poly.get_area(), reverse=True)
        self.perimeter = self.get_perimeter()

        log.info(f"Complex polygon initialized, total area= {self.area}, total perimeter= {self.perimeter}")
        log.debug(f"Number of inner holes= {len(self.get_hole_polygons())} number of outer polygons= {len(self.get_outer_polygons())}")
        log.debug(f"Areas of outer polygons: {[poly.get_area(absolute=False) for poly in self.polygons if poly.get_orientation() == polygon_type.OUTER]} ")
        log.debug(f"Areas of hole polygons: {[poly.get_area(absolute=False) for poly in self.polygons if poly.get_orientation() == polygon_type.HOLE]} ")
    
    def get_area(self) -> float:
        """
        Calculates and returns the total area of the polygons contained within the object.
        If the area has not been previously computed, it iterates through each polygon,
        checks if the polygon's signed area is available, and adds the (possibly negative)
        area to the total. The computed area is cached for future calls.
        Returns:
            float: The total area of all polygons.
        """

        if self.area is None:
            total_area = 0
            for polygon in self.polygons:
                if polygon._signed_area is not None:
                    total_area += polygon.get_area(absolute=False)* -1
            self.area = total_area
        
        return self.area
    
    def get_perimeter(self) -> float:
        """
        Calculates and returns the total perimeter of all polygons in the collection.
        If the perimeter has not been previously calculated, it sums the perimeters of each polygon
        in `self.polygons` and caches the result in `self.perimeter`. Subsequent calls return the cached value.
        Returns:
            float: The total perimeter of all polygons.
        """

        if self.perimeter is None:
            total_perimeter = 0
            for polygon in self.polygons:
                total_perimeter += polygon.get_perimeter()
            self.perimeter = total_perimeter
        
        return self.perimeter

    def get_outer_polygons(self) -> list:
        """
        Retrieves the vertices of the outer polygons from the collection.
        Iterates through the polygons in the instance, selecting those whose orientation
        is classified as OUTER. Returns the vertices of the first outer polygon found,
        or an empty list if none are present.
        Returns:
            list: Vertices of the first outer polygon, or an empty list if none exist.
        """

        outerps = [poly.vertices for poly in self.polygons if poly.get_orientation() == polygon_type.OUTER]
        return outerps[0] if len(outerps) > 0 else []

    def get_hole_polygons(self) -> list:
        """
        Returns a list of vertices for all polygons classified as holes.
        Iterates through the collection of polygons and selects those whose orientation
        matches the HOLE type, returning their vertices.
        Returns:
            list: Vertices of the first inner polygons, or an empty list if none exist.
        """

        return [poly.vertices for poly in self.polygons if poly.get_orientation() == polygon_type.HOLE]

    def get_all_polygons(self) -> list:
        return self.get_outer_polygons() + self.get_hole_polygons()


    def rotate(self, angle_deg: float, origin=(0, 0)):        
        """
        Rotates all polygons in the collection by a specified angle around a given origin.
        Args:
            angle_deg (float): The angle in degrees to rotate the polygons.
            origin (tuple, optional): The (x, y) coordinates of the rotation origin. Defaults to (0, 0).
        Notes:
            Only polygons of type `simple_polygon` are rotated. Other types are ignored.
        """
        
        for poly in self.polygons:
            if isinstance(poly, simple_polygon):
                poly.rotate(angle_deg=angle_deg, origin=origin)

    def _plot_add_polygon(self, polygon:"simple_polygon", plot_vertices=False, polygon_legend=None, _plt:plt=plt) -> plt:
        """
        Adds a polygon to a matplotlib plot, optionally plotting its vertices and setting a legend.
        Parameters:
            polygon (simple_polygon): The polygon object to plot. Must be an instance of simple_polygon.
            plot_vertices (bool, optional): If True, plot the vertices of the polygon as red points. Defaults to False.
            polygon_legend (str, optional): Legend label for the polygon. If None, defaults to "Polygon".
            _plt (matplotlib.pyplot, optional): Matplotlib pyplot object to plot on. Defaults to plt.
        Returns:
            matplotlib.pyplot: The updated matplotlib pyplot object with the polygon added.
        Notes:
            - Outer polygons are filled with a random color and labeled.
            - Hole polygons are filled with white and their vertices are reversed for correct orientation.
            - If the input polygon is not an instance of simple_polygon, the plot is returned unchanged.
        """

        if not isinstance(polygon, simple_polygon):
            return _plt
        
        orientation = polygon.get_orientation()
        vertices = np.array(polygon.vertices)
    
        _plt.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2)
        
        if orientation == polygon_type.OUTER: 
            color = np.random.rand(3,)  # random RGB in [0,1]
            _plt.fill(vertices[:, 0], vertices[:, 1], color=color, label=polygon_legend if polygon_legend is not None else "Polygon")
        elif orientation == polygon_type.HOLE: 
            vertices = vertices[::-1]
            _plt.fill(vertices[:, 0], vertices[:, 1], color='white')
        
        if plot_vertices:
            _plt.scatter(vertices[:, 0], vertices[:, 1], color='red')
        return _plt

    def _get_polygon_plot(self, plot_vertices=False, title_legend="Polygon", x_axis_legend="X", y_axis_legend="Y") -> plt:
        """
        Generates a matplotlib plot of the polygons stored in the object.
        Args:
            plot_vertices (bool, optional): If True, plot the vertices of each polygon. Defaults to False.
            title_legend (str, optional): Title of the plot. Defaults to "Polygon".
            x_axis_legend (str, optional): Label for the X axis. Defaults to "X".
            y_axis_legend (str, optional): Label for the Y axis. Defaults to "Y".
        Returns:
            matplotlib.pyplot: The matplotlib pyplot object with the plotted polygons.
        """

        plt.figure(figsize=(5,5))

        for polygon in self.polygons:
            # Close the polygon by repeating the first point
            #polygon_closed = np.vstack([polygon, polygon[0]]
            self._plot_add_polygon(polygon, plot_vertices=plot_vertices, _plt=plt)
            plt.title(title_legend)
            plt.xlabel(x_axis_legend)
            plt.ylabel(y_axis_legend)
            plt.axis("equal")
        return plt
    
    def plot_polygon(self, plot_vertices=False, title_legend="Polygon", x_axis_legend="X", y_axis_legend="Y", save_path: str = None):
        """
        Plots the polygon using matplotlib, with optional display of vertices and customizable legends.
        Args:
            plot_vertices (bool, optional): If True, plot the vertices of the polygon. Defaults to False.
            title_legend (str, optional): Title for the plot legend. Defaults to "Polygon".
            x_axis_legend (str, optional): Label for the X axis. Defaults to "X".
            y_axis_legend (str, optional): Label for the Y axis. Defaults to "Y".
            save_path (str, optional): If provided, saves the plot to the specified file path. Defaults to None.
        Side Effects:
            Displays the plot window.
            Saves the plot to a file if save_path is specified.
        """
        
        plt = self._get_polygon_plot(plot_vertices=plot_vertices, title_legend=title_legend, x_axis_legend=x_axis_legend, y_axis_legend=y_axis_legend)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log.info(f"Polygon plot saved to {save_path}")
        plt.show()

    def _extract_rims_from_shapely(self, poly: Polygon) -> list:
        """Extract outer and holes from a shapely polygon as numpy arrays."""
        #print(f"number of exterior rings: {len(poly.exterior)}")
        outer = np.array(poly.exterior.coords)
        holes = [np.array(ring.coords) for ring in poly.interiors]
        return [outer] + holes

    def _polygon_op(self, poly1: Polygon, poly2: Polygon, _op:_OP_TYPE =Polygon.union) -> Polygon:
        """Apply a Shapely operation (union or intersection) between two polygons."""
        return _op(poly1, poly2)

    def unify_polygons(self, poly2: "complex_polygon", _op=Polygon.union) -> Union["complex_polygon", "multi_complex_polygon"]:
        """
        Unifies two complex polygons using a specified geometric operation (default is union).
        This method takes another `complex_polygon` object and applies the given operation
        (such as union) to combine the polygons, including their holes and outer boundaries.
        It prints diagnostic information about the input polygons, including area, perimeter,
        and counts of holes and outer polygons.
        Args:
            poly2 (complex_polygon): The second complex polygon to unify with the current one.
            _op (callable, optional): The geometric operation to apply (default: Polygon.union).
        Returns:
            Union[complex_polygon, multi_complex_polygon]: 
                - If the result is empty, returns a single empty `complex_polygon` object.
                - If the result is a single polygon, returns a `complex_polygon` representing the unified shape.
                - If the result is a multipolygon, returns a `multi_complex_polygon` containing all resulting polygons.
        """

        log.info(f"Polygon 1 area: {self.get_area()}, perimeter: {self.get_perimeter()}")
        log.debug(f"Number of inner holes: {len(self.get_hole_polygons())} Number of outer polygons: {len(self.get_outer_polygons())}")
        p1 = Polygon(shell=self.get_outer_polygons(), holes=self.get_hole_polygons())
        
        log.info(f"Polygon 2 area: {poly2.get_area()}, perimeter: {poly2.get_perimeter()}")
        log.debug(f"Number of inner holes: {len(poly2.get_hole_polygons())} Number of outer polygons: {len(poly2.get_outer_polygons())}")
        p2 = Polygon(shell=poly2.get_outer_polygons(), holes=poly2.get_hole_polygons())
        
        #union = p1.union(p2)
        rest_op = self._polygon_op(p1, p2, _op=_op)

        if rest_op.is_empty:
            # No intersection: return original polygons as one complex polygon
            return complex_polygon(self.get_all_polygons()+poly2.get_all_polygons())
        else:
            # Intersection can be polygon or multipolygon
            if rest_op.geom_type == "Polygon":
                return complex_polygon(self._extract_rims_from_shapely(rest_op))
            elif rest_op.geom_type == "MultiPolygon":
                print(f"Result is a MultiPolygon with {len(rest_op.geoms)} geometries.")
                multi_poly = []
                for geom in rest_op.geoms:
                    multi_poly.append(complex_polygon(self._extract_rims_from_shapely(geom)))
                return multi_complex_polygon(multi_poly)
        return complex_polygon()

    def intersect_polygons(self, poly2: "complex_polygon") -> Union["complex_polygon", "multi_complex_polygon"]:
        """
        Computes the intersection between this polygon and another polygon.
        Args:
            poly2 (complex_polygon): The polygon to intersect with.
        Returns:
            Union[complex_polygon, multi_complex_polygon]: The resulting polygon(s) from the intersection.
        """

        return self.unify_polygons(poly2, _op=Polygon.intersection)
    
    def apply_polygon_simplification(self, collinear_tol=1e-6, distance_threshold=1e-2):
        """
        Simplifies each polygon in the collection by reducing collinear points and merging close vertices.
        Args:
            collinear_tol (float, optional): Tolerance for detecting collinear points. Defaults to 1e-6.
            distance_threshold (float, optional): Minimum distance between vertices to consider merging. Defaults to 1e-2.
        Notes:
            - Only polygons of type `simple_polygon` are processed.
            - The simplification is applied in-place to each polygon.
        """
        
        for poly in self.polygons:
            if isinstance(poly, simple_polygon):
                poly.apply_polygon_simplification(collinear_tol==collinear_tol, distance_threshold=distance_threshold)

    def read_polygon_from_npz_file(self, file_path: str):
        """
        Reads polygon data from a NumPy .npz file and reinitializes the object with the loaded polygons.
        Args:
            file_path (str): The path to the .npz file containing polygon arrays.
        Loads:
            The method expects the .npz file to contain arrays named in the format 'arr_{i}'.
            It loads all arrays into a list and reinitializes the object with this list.
        Prints:
            The type and number of polygons loaded, along with the source file path.
        """

        loaded = np.load(file_path)
        read_polygon_arrays = [loaded[f"arr_{i}"] for i in range(len(loaded.files))]
        log.info(f"Loaded in type: {type(read_polygon_arrays)} with {len(read_polygon_arrays)} polygons from {file_path}")
        self.__init__(read_polygon_arrays)
    
    def export_to_npz_file(self, file_path: str="./output/polygon_data.npz"):
        """
        Exports the polygon data to a NumPy .npz file.
        Args:
            file_path (str): The path where the .npz file will be saved.
        Saves:
            The method saves each polygon's vertices as separate arrays in the .npz file,
            named in the format 'arr_{i}'.
        Side Effects:
            Creates the directory for the file if it does not exist.
            Logs an informational message upon successful saving.
        """

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez(file_path, **{f"arr_{i}": poly.vertices for i, poly in enumerate(self.polygons)})
        log.info(f"Polygon data exported to {file_path}")

class multi_complex_polygon:
    def __init__(self, complex_polygons: list=[]):
        self.complex_polygons = [cpoly for cpoly in complex_polygons if isinstance(cpoly,complex_polygon)] if complex_polygons is not None else []
    
    def get_area(self) -> float:
        """
        Calculates and returns the total area of all complex polygons contained within the object.
        Iterates over the list of complex polygons (`self.complex_polygons`), checks if each item is an instance
        of `complex_polygon`, and sums their areas by calling their `get_area()` method.
        Returns:
            float: The total area of all valid complex polygons.
        """

        return sum([cpoly.get_area() for cpoly in self.complex_polygons if isinstance(cpoly, complex_polygon)])
    
    def get_perimeter(self) -> float:
        """
        Calculates and returns the total perimeter of all complex polygons contained within the object.
        Iterates through the list of complex polygons (`self.complex_polygons`), checks if each element is an instance of `complex_polygon`,
        and sums their perimeters by calling their `get_perimeter()` method.
        Returns:
            float: The total perimeter of all valid complex polygons.
        """

        return sum([cpoly.get_perimeter() for cpoly in self.complex_polygons if isinstance(cpoly, complex_polygon)])
    
    def plot_polygon(self, plot_vertices=False, title_legend="Polygon", x_axis_legend="X", y_axis_legend="Y", save_path: str = None):
        """
        Plots the polygons contained in the `complex_polygons` attribute.
        Args:
            plot_vertices (bool, optional): If True, plot the vertices of each polygon. Defaults to False.
            title_legend (str, optional): Title for the plot. Defaults to "Polygon".
            x_axis_legend (str, optional): Label for the X axis. Defaults to "X".
            y_axis_legend (str, optional): Label for the Y axis. Defaults to "Y".
            save_path (str, optional): If provided, saves the plot to the specified file path. Defaults to None.
        Notes:
            - Each polygon is plotted with its vertices optionally highlighted.
            - The plot is displayed using matplotlib and can be saved to a file if `save_path` is specified.
            - The plot axes are set to equal aspect ratio.
        """

        #plt.figure(figsize=(5,5))
        for i,cpoly in enumerate(self.complex_polygons):
            for polygon in cpoly.polygons:
                if not isinstance(cpoly, complex_polygon):
                    continue
                # Close the polygon by repeating the first point
                #polygon_closed = np.vstack([polygon, polygon[0]]
                cpoly._plot_add_polygon(polygon, plot_vertices=plot_vertices, polygon_legend=f"Polygon {i+1}", _plt=plt)
                plt.title(title_legend)
                plt.xlabel(x_axis_legend)
                plt.ylabel(y_axis_legend)
                plt.legend()
                plt.axis("equal")
    
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            log.info(f"Polygon plot saved to {save_path}")
        
        plt.show()
    
    def get_polygon(self, index: int =0) -> complex_polygon | None:
        """
        Retrieves a complex polygon from the list by its index.
        Args:
            index (int): The index of the polygon to retrieve.
        Returns:
            complex_polygon | None: The complex polygon at the specified index if it exists,
            otherwise None.
        """

        if 0 <= index < len(self.complex_polygons):
            return self.complex_polygons[index]
        return None

    def export_to_npz_file(self, file_path: str="./output/multi_polygon_data.npz"):
        """
        Exports all complex polygons in the collection to individual NPZ files.

        Each polygon is saved as a separate NPZ file in the specified directory.
        The directory is created if it does not exist. The filenames are generated
        as 'poly_{i}.npz', where 'i' is the index of the polygon in the collection.

        Args:
            file_path (str): Path to the output directory or NPZ file. Defaults to
                './output/multi_polygon_data.npz'. The directory portion of this path
                is used for saving the files.

        Side Effects:
            Creates the output directory if it does not exist.
            Writes NPZ files for each complex polygon in the collection.
            Logs the export operation.
        """

        file_path = os.path.dirname(file_path) + "/"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        for i, cpoly in enumerate(self.complex_polygons):
            cpoly.export_to_npz_file(file_path=file_path+f"/poly_{i}.npz")
        log.info(f"Multi-complex polygon data exported to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='A simple program that processes 2 polygon files (npz) and applied either union or interserction')
    
    parser.add_argument('-p1', '--poly1', help='Specify the first polygon file (npz)', required=True)
    parser.add_argument('-p2', '--poly2', help='Specify the second polygon file (npz)', required=True)
    parser.add_argument('-o', '--output', help='Specify an output directory to save the results', default="./output")
    parser.add_argument('-of', '--output_file', help='Specify an output file name to save the results', default=None)
    parser.add_argument('-op', '--operation', help='Specify the operation to perform: union or intersection', choices=['union', 'intersection'], default='union')
    parser.add_argument('-s', '--simplify', help='Apply polygon simplification before operation', action='store_true')
    parser.add_argument('-dt', '--distance_threshold', help='Distance threshold for polygon simplification', type=float, default=1e-0)
    parser.add_argument('-ct', '--collinear_tol', help='Collinear tolerance for polygon simplification', type=float, default=1e-4)

    args = parser.parse_args()
    if not os.path.isfile(args.poly1):
        log.error(f"Polygon file 1 does not exist: {args.poly1}")
        return
    if not os.path.isfile(args.poly2):
        log.error(f"Polygon file 2 does not exist: {args.poly2}")
        return
    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    poly1 = complex_polygon()
    poly1.read_polygon_from_npz_file(args.poly1)
    
    poly2 = complex_polygon()
    poly2.read_polygon_from_npz_file(args.poly2)

    if args.simplify:
        poly1.apply_polygon_simplification(collinear_tol=args.collinear_tol, distance_threshold=args.distance_threshold)
        poly2.apply_polygon_simplification(collinear_tol=args.collinear_tol, distance_threshold=args.distance_threshold)
    poly1.plot_polygon(plot_vertices=True, title_legend="Polygon 1")
    poly2.plot_polygon(plot_vertices=True, title_legend="Polygon 2")

    result = complex_polygon()
    if args.operation == 'union':
        result = poly1.unify_polygons(poly2)
    elif args.operation == 'intersection':
        result = poly1.intersect_polygons(poly2)
    result.plot_polygon(title_legend=f"{args.operation} of Polygon 1 and Polygon 2", save_path=os.path.join(args.output, f"{args.operation}_polygon1_polygon2.png"))

    if args.output_file is not None:
        output_file_path = os.path.join(args.output, args.output_file)
        result.export_to_npz_file(file_path=output_file_path)

if __name__ == "__main__":
    main()   