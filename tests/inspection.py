from thesis_assyrian_relief.utils.inspection import (
    get_relief_rows,
    plot_relief_views,
    plot_query_and_retrieved_reliefs,
)

image_level_csv = "data/splits/image_level_dataset.csv"
image_root = "/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset"

rows = get_relief_rows(image_level_csv, "BM 124773")
print(rows[["Relief_ID", "Authority", "view_index", "suffix"]])

fig1 = plot_relief_views(
    image_level_csv=image_level_csv,
    image_root=image_root,
    relief_id="BM 124773",
    show=False,
)
fig1.savefig("outputs/test_bm124773_views.png", dpi=150, bbox_inches="tight")

fig2 = plot_query_and_retrieved_reliefs(
    image_level_csv=image_level_csv,
    image_root=image_root,
    query_relief_id="BM 124773",
    retrieved_relief_ids=["BM 136774", "BM 124794"],
    show=False,
)
fig2.savefig("outputs/test_bm124773_neighbors.png", dpi=150, bbox_inches="tight")

print("Saved inspection figures.")