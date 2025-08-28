from unite_polygons import complex_polygon, multi_complex_polygon
import numpy as np

def main():
    '''1.- Polygon #1 is manually defined and slightly rotated'''
    manual_poly = [[24,42], [30,42], [24,75], [24,42]]
    manual_poly = np.array(manual_poly[::-1]) # Reverse to make it clockwise
    poly1 = complex_polygon([manual_poly])
    poly1.rotate(angle_deg=3, origin=(24,42))
    poly1.plot_polygon(title_legend="Polygon 1", save_path="./output_example/polygon1.png")

    '''2.- A complex polygon #2 from file is loaded here'''
    poly2 = complex_polygon()
    poly2.read_polygon_from_npz_file("./NumpyFiles/arrays_90.npz")
    poly2.plot_polygon(plot_vertices=True, title_legend="Polygon 2", save_path="./output_example/polygon2.png")

    '''3.- Polygon #2 is simplified to reduce number of vertices'''
    poly2.apply_polygon_simplification(distance_threshold=1e-0)
    poly2.plot_polygon(plot_vertices=True, title_legend="Polygon 2 after simplification", save_path="./output_example/polygon2_simplified.png")

    '''4.- Union of both polygons #1 and #2 is computed and plotted'''
    result = poly1.unify_polygons(poly2)
    print(f"Resulting Polygon area: {result.get_area()}, perimeter: {result.get_perimeter()}")
    # union of two polugons can result in multiple polygons
    result.plot_polygon(title_legend="Union of Polygon 1 and 2", save_path="./output_example/union_polygon1_polygon2.png")
    if isinstance(result, multi_complex_polygon):
        retult = result.get_polygon(0)

    '''5.- Intersection of both polygons is computed as well'''
    result_int = poly1.intersect_polygons(poly2)
    print(f"Resulting Polygon area: {result.get_area()}, perimeter: {result.get_perimeter()}")
    result_int.plot_polygon(title_legend="Intersection of Polygon 1 and 2", save_path="./output_example/intersection_polygon1_polygon2.png")
    if isinstance(result_int, multi_complex_polygon):
        result_int = result_int.get_polygon(0)

    '''6.- A third polygon #3 is created from manual definition and rotated'''
    poly3 = complex_polygon([manual_poly])
    poly3.rotate(angle_deg=-25, origin=(24,42))
    poly3.plot_polygon(title_legend="Polygon 3")

    '''7.- Union of polygon #3 with the intersection of polygon #1 and #2 is computed and plotted'''
    if result_int is None:
        return
    
    poly3.unify_polygons(result_int).plot_polygon(title_legend="Union of Polygon 3 and Intersection of Polygon 1 and 2 (comp 1)", save_path="./output_example/union_polygon3_intersection_polygon1_polygon2.png")
    print(f"Resulting Polygon 3 area: {poly3.get_area()}, perimeter: {poly3.get_perimeter()}")



if __name__ == "__main__":
    main()