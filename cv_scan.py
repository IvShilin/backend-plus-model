import cv2
import numpy as np
from constants import *


def process_image(image_data: np.ndarray) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    binary_image = cv2.threshold(blurred_image, 100, 150, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contour_list = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    largest_contour = max(contour_list, key=cv2.contourArea)
    corner_points = cv2.approxPolyDP(largest_contour, 0.1 * cv2.arcLength(largest_contour, True), True)

    corner_points = corner_points.reshape(-1, 2)
    center = np.mean(corner_points, axis=0)
    angles = np.arctan2(corner_points[:, 1] - center[1], corner_points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    corner_points = corner_points[sorted_indices]

    src_pts = np.float32(corner_points[:4])
    width = max(
        np.linalg.norm(src_pts[0] - src_pts[1]),
        np.linalg.norm(src_pts[2] - src_pts[3]),
    )
    height = max(
        np.linalg.norm(src_pts[0] - src_pts[3]),
        np.linalg.norm(src_pts[1] - src_pts[2]),
    )
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_image = cv2.warpPerspective(image_data, matrix, (int(width), int(height)))

    corrected_image = cv2.addWeighted(corrected_image, 1.5, cv2.GaussianBlur(corrected_image, (5, 5), 0), -0.5, 0)
    corrected_image = cv2.bilateralFilter(corrected_image, 5, 50, 50)
    corrected_image_gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    corrected_image = cv2.adaptiveThreshold(corrected_image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    corrected_image = cv2.fastNlMeansDenoising(corrected_image, None, 10, 7, 21)

    return corrected_image


def apply_filter(image: np.ndarray) -> np.ndarray:
    filtered_image = cv2.medianBlur(image, 5)
    return filtered_image


def scan_and_mark_cells(image_data: np.ndarray) -> tuple:
    filtered_data = apply_filter(image_data)
    grayscale_data = cv2.cvtColor(filtered_data, cv2.COLOR_BGR2GRAY)
    _, binary_data = cv2.threshold(grayscale_data, 140, 240, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contour_list, _ = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_categories = {}
    for contour in contour_list:
        x_coord, y_coord, _, _ = cv2.boundingRect(contour)
        if x_coord < 5:
            contour_categories.setdefault('left_column', []).append(contour)
        elif y_coord < 5:
            contour_categories.setdefault('top_row', []).append(contour)
        else:
            contour_categories.setdefault('other_cells', []).append(contour)

    marked_data = image_data.copy()
    for contour in contour_categories['other_cells']:
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        cv2.rectangle(marked_data, (x_coord, y_coord), (x_coord + width, y_coord + height), (0, 255, 0), 2)

    return contour_categories['other_cells'], contour_categories['left_column'], contour_categories['top_row']


def enhance_img(image: np.ndarray) -> np.ndarray:
    img = cv2.convertScaleAbs(image, alpha=ALPHA, beta=BETA)
    img = cv2.filter2D(img, -1, SHARP)
    return img


def find_empty_cells(image: np.ndarray, contours: list) -> list:
    img = image.copy()
    img = enhance_img(img)
    empty_cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = img[y: y + h, x: x + w]
        mean_intensity = np.mean(roi)
        if mean_intensity < THRESHOLD:
            empty_cells.append(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return empty_cells


def filter_out_small_contours(contour_list: list, min_area_threshold: int) -> list:
    return [
        contour for contour in contour_list if cv2.contourArea(contour) > min_area_threshold
    ]


def remove_similar_contour_shapes(contour_list: list, proximity_threshold: int) -> list:
    filtered_contour_list = []
    for contour in contour_list:
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        is_close_to_existing = False
        for existing_contour in filtered_contour_list:
            existing_x, existing_y, _, _ = cv2.boundingRect(existing_contour)
            if abs(y_coord - existing_y) < proximity_threshold:
                is_close_to_existing = True
                break
        if not is_close_to_existing:
            filtered_contour_list.append(contour)
    return filtered_contour_list


def calculate_contour_centroids(contour_list: list) -> list:
    centroid_list = []
    for contour in contour_list:
        x_coord, y_coord, width, height = cv2.boundingRect(contour)
        centroid_x = x_coord + width // 2
        centroid_y = y_coord + height // 2
        centroid_list.append((centroid_x, centroid_y))
    return centroid_list


def find_solution(
        contour_points: list,
        column_contour_map: dict,
        row_contour_map: dict,
        correct_solutions: list
) -> dict:
    solution_set = set()
    for point in contour_points:
        for row_key, row_contour in row_contour_map.items():
            x_row, y_row, w_row, h_row = cv2.boundingRect(row_contour)
            if y_row <= point[1] <= y_row + h_row:
                for col_key, col_contour in column_contour_map.items():
                    x_col, y_col, w_col, h_col = cv2.boundingRect(col_contour)
                    if x_col <= point[0] <= x_col + w_col:
                        solution_set.add((row_key, col_key))

    result_dict = {"solutions": [], "correct_count": 0, "incorrect_count": 0}
    for row, col in solution_set:
        for correct_solution in correct_solutions:
            if row == correct_solution["question"]:
                correct_answer = correct_solution["correct_answer"]
                result_dict["solutions"].append({"question": row, "answer": col, "correct_answer": correct_answer})
                result_dict["correct_count" if col == correct_answer else "incorrect_count"] += 1
                break
        else:
            result_dict["solutions"].append({"question": row, "answer": col, "correct_answer": None})
            result_dict["incorrect_count"] += 1

    return result_dict


def find_solution(
        contour_points: list,
        column_contour_map: dict,
        row_contour_map: dict,
        correct_solutions: list
) -> dict:
    solution_set = set()
    for point in contour_points:
        for row_key, row_contour in row_contour_map.items():
            x_row, y_row, w_row, h_row = cv2.boundingRect(row_contour)
            if y_row <= point[1] <= y_row + h_row:
                for col_key, col_contour in column_contour_map.items():
                    x_col, y_col, w_col, h_col = cv2.boundingRect(col_contour)
                    if x_col <= point[0] <= x_col + w_col:
                        solution_set.add((row_key, col_key))

    result_dict = {"solutions": [], "correct_count": 0, "incorrect_count": 0}
    for row, col in solution_set:
        for correct_solution in correct_solutions:
            if row == correct_solution["question"]:
                correct_answer = correct_solution["correct_answer"]
                result_dict["solutions"].append({"question": row, "answer": col, "correct_answer": correct_answer})
                result_dict["correct_count" if col == correct_answer else "incorrect_count"] += 1
                break
        else:
            result_dict["solutions"].append({"question": row, "answer": col, "correct_answer": None})
            result_dict["incorrect_count"] += 1

    return result_dict


def process_image_and_find_solution(image_path, correct_answers):
    image_data = cv2.imread(image_path)

    try:
        processed_image = process_image(image_data)
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)

    contours, _ = cv2.findContours(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contours = [contour for contour in contours if cv2.contourArea(contour) > AREA]

    row_contour_map = {str(i + 1): contour for i, contour in enumerate(contours[:7])}
    column_contour_map = {chr(i + 65): contour for i, contour in enumerate(contours[7:14])}

    centroid_points = calculate_contour_centroids(find_empty_cells(processed_image, contours[14:]))

    result = find_solution(centroid_points, column_contour_map, row_contour_map, correct_answers)

    return result

