"""Functions for RPA tasks, i.e. searching locations in documents."""

import matplotlib.pyplot as plt
from loguru import logger
from rapidfuzz import fuzz

from .types import (
    BaseRPAProcessor,
    OCRResult,
    PageResult,
    PositionalQuery,
    SearchResult,
    SemanticQuery,
)


class RPAProcessor(BaseRPAProcessor):
    """RPA processor for searching text in OCR results."""

    def __init__(
        self,
        search_type: str,
        search_kwargs: dict | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        """Initialise the RPA processor."""
        self.search_type = search_type
        self.search_kwargs = search_kwargs if search_kwargs is not None else {}
        self.verbose = verbose

        if self.search_type == "semantic":
            # Here implement semantic search initialisation if needed i.e loading models
            pass
        elif self.search_type == "positional":
            pass
        else:
            error_message = f"Unsupported search type: {search_type}"
            raise ValueError(error_message)

    def _compute_semantic_similarity(
        self, text1: str, text2: str, method: str
    ) -> float:
        """
        Compute semantic similarity between two texts using specified method.

        Args:
            text1: First text string
            text2: Second text string
            method: Similarity method to use ("fuzzy" supported currently)
            Returns:"
        Returns:
            Similarity score between 0 and 1

        """
        if method == "fuzzy":
            # TO DO: Explore other fuzzy matching methods
            score = fuzz.partial_ratio(text1.lower(), text2.lower())
            if self.verbose:
                logger.info(
                    f"Fuzzy matching score between '{text1}' and '{text2}': {score}"
                )
            return score / 100.0  # Normalize to [0, 1]
        error_message = f"Unsupported semantic similarity method: {method}"
        raise ValueError(error_message)

    def _rectangle_rectangle_intersection(
        self, rect1: list[list[float]], rect2: list[list[float]], rect1_pad: float = 0
    ) -> bool:
        """
        Check if two rectangles intersect, with optional padding on rect1.

        Args:
            rect1: First rectangle defined by 4 corner list of [x, y] points
            rect2: Second rectangle defined by 4 corner list of [x, y] points
            rect1_pad: Padding to expand rect1 by (in pixels)

        Returns:
            True if rectangles intersect, False otherwise

        """
        # first expand rect1 by rect1_pad
        rect1_x1 = min([point[0] for point in rect1]) - rect1_pad
        rect1_x2 = max([point[0] for point in rect1]) + rect1_pad
        rect1_y1 = min([point[1] for point in rect1]) - rect1_pad
        rect1_y2 = max([point[1] for point in rect1]) + rect1_pad

        rect2_x1 = min([point[0] for point in rect2])
        rect2_x2 = max([point[0] for point in rect2])
        rect2_y1 = min([point[1] for point in rect2])
        rect2_y2 = max([point[1] for point in rect2])

        # check for no overlap
        return not (
            rect1_x1 > rect2_x2
            or rect2_x1 > rect1_x2
            or rect1_y1 > rect2_y2
            or rect2_y1 > rect1_y2
        )

    def _circle_rectangle_intersection(
        self, circle_x: float, circle_y: float, radius: float, rect: list[list[float]]
    ) -> bool:
        """
        Check if a circle intersects with a rectangle.

        Args:
            circle_x: X coordinate of circle center
            circle_y: Y coordinate of circle center
            radius: Radius of the circle
            rect: Rectangle defined by 4 corner list of [x, y] points
        Returns:
            True if circle intersects rectangle, False otherwise.

        """
        rect_x1 = min([point[0] for point in rect])
        rect_x2 = max([point[0] for point in rect])
        rect_y1 = min([point[1] for point in rect])
        rect_y2 = max([point[1] for point in rect])

        # Check if circle center is inside rectangle
        if (
            circle_x >= rect_x1
            and circle_x <= rect_x2
            and circle_y >= rect_y1
            and circle_y <= rect_y2
        ):
            return True

        # Now check for intersection
        # closest x, y on the rectangle to the circle center
        closest_x = max(rect_x1, min(circle_x, rect_x2))
        closest_y = max(rect_y1, min(circle_y, rect_y2))

        # distance from circle center to closest point
        distance_x = circle_x - closest_x
        distance_y = circle_y - closest_y

        distance_squared = (distance_x**2) + (distance_y**2)
        return distance_squared <= (radius**2)

    def _debug_search_results(
        self,
        page: PageResult,
        query: PositionalQuery | SemanticQuery,
        found_results: list[OCRResult],
    ) -> None:
        """Debug visualisation showing all found bounding boxes on the original image."""
        if page.original_image is None:
            logger.warning("No original image available for debugging visualisation")
            return

        fig, ax = plt.subplots(figsize=(15, 10))

        # Display the original image
        if len(page.original_image.shape) == 3:  # noqa: PLR2004
            # Convert BGR to RGB for matplotlib
            image_rgb = page.original_image[:, :, ::-1]
            ax.imshow(image_rgb)
        else:
            ax.imshow(page.original_image, cmap="gray")

        # Draw search area (for positional search)
        if isinstance(query, PositionalQuery):
            circle = plt.Circle(
                (query.x, query.y),
                query.search_radius,
                color="red",
                fill=False,
                linewidth=3,
                alpha=0.8,
                label="Search Area",
            )
            ax.add_patch(circle)

            # Mark query center
            ax.plot(
                query.x,
                query.y,
                "ro",
                markersize=10,
                label=f"Query Center ({query.x:.0f}, {query.y:.0f})",
            )

        # Draw all OCR results on the page
        for ocr_result in page.data:
            poly = ocr_result.poly

            # Check if this OCR result was found in our search
            is_match = ocr_result in found_results

            # Draw polygon bounding box
            if len(poly) >= 3:  # noqa: PLR2004
                poly_x = [point[0] for point in poly] + [
                    poly[0][0]
                ]  # Close the polygon
                poly_y = [point[1] for point in poly] + [poly[0][1]]

                if is_match:
                    # Highlight matches in bright green
                    ax.plot(poly_x, poly_y, "lime", linewidth=3, alpha=0.9)
                    ax.fill(poly_x, poly_y, "lime", alpha=0.3)

                else:
                    # Draw non-matches in light blue
                    ax.plot(poly_x, poly_y, "lightblue", linewidth=1, alpha=0.6)

        # Add title with search info
        if isinstance(query, PositionalQuery):
            title = "Positional Search Results\n"
            title += f"Query: ({query.x:.0f}, {query.y:.0f}), Radius: {query.search_radius:.0f}px\n"
        else:
            title = "Semantic Search Results\n"
            title += f'Query: "{getattr(query, "text", "N/A")}"\n'

        ax.set_title(title, fontsize=12, pad=20)

        # Add legend
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1))
        ax.axis("off")  # Hide axes for cleaner visualisation

        # Add summary text box
        summary_text = f"Page {page.page}\n"
        summary_text += f"Total OCR regions: {len(page.data)}\n"
        summary_text += f"Matches found: {len(found_results)}\n"
        summary_text += f"Search type: {self.search_type}"

        ax.text(
            0.02,
            0.02,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
        )

        plt.tight_layout()
        plt.show()

        # Log the results
        logger.info(
            f"Search visualization complete: {len(found_results)} matches found"
        )
        for i, result in enumerate(found_results):
            logger.info(
                f"  Match {i+1}: '{result.text}' (confidence: {result.score:.3f})"
            )

    def _postional_search(
        self, page: PageResult, query: PositionalQuery
    ) -> list[OCRResult]:
        """
        Perform positional search for a single query.

        Args:
            page: PageResult containing OCR data
            query: PositionalQuery with x, y, and search_radius
        Returns:
            List of OCRResult that match the positional query

        """
        matches = []

        for ocr_result in page.data:
            # Check for intersection between query circle and OCR entry bounding box
            if self._circle_rectangle_intersection(
                query.x, query.y, query.search_radius, ocr_result.poly
            ):
                matches.append(ocr_result)  # noqa: PERF401

        return matches

    def _semantic_search(
        self, page: PageResult, query: SemanticQuery
    ) -> list[OCRResult]:
        """
        Perform semantic search for a single query.

        Args:
            page: PageResult containing OCR data
            query: SemanticQuery with text and threshold
        Returns:
            List of OCRResult that match the semantic query

        """
        semantic_matches = []
        for ocr_result in page.data:
            score = self._compute_semantic_similarity(
                ocr_result.text, query.text, query.search_type
            )
            if score >= query.threshold:
                logger.info(
                    f"Semantic match found: '{query.text}' match '{ocr_result.text}' with score of {score:.3f}"
                )
                semantic_matches.append(ocr_result)

        # get neighbouring boxes based on search padding
        matches = []
        for ocr_result in page.data:
            for sem_match in semantic_matches:
                if self._rectangle_rectangle_intersection(
                    sem_match.poly, ocr_result.poly, rect1_pad=query.search_padding
                ):
                    matches.append(ocr_result)
                    break  # no need to check other semantic matches

        return matches

    def search(
        self,
        page: PageResult,
        query: PositionalQuery | SemanticQuery,
        task: str | None = None,
    ) -> SearchResult:
        """
        Search for queries in the provided OCR results.

        Args:
            page: PageResult containing OCR data
            query: PositionalQuery or SemanticQuery
        Returns:
            SearchResult containing matched OCR results and metadata

        """
        if self.search_type == "semantic":
            if not isinstance(query, SemanticQuery):
                error_message = "Semantic search requires a SemanticQuery"
                raise TypeError(error_message)
            logger.info("Performing semantic search...")
            results = self._semantic_search(page, query)
        elif self.search_type == "positional":
            if not isinstance(query, PositionalQuery):
                error_message = "Positional search requires a PositionalQuery"
                raise TypeError(error_message)
            logger.info("Performing positional search...")
            results = self._postional_search(page, query)

        result_page = PageResult(
            page=page.page,
            data=results,
            original_image=page.original_image,
        )

        search_result = SearchResult(
            page_result=result_page,
            search_type=self.search_type,
            search_task=task,
        )
        if self.verbose:
            self._debug_search_results(page, query, results)

        return search_result
