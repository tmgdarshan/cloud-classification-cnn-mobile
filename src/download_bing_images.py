from bing_image_downloader import downloader

# Download 100 images of cirrus clouds
downloader.download(
    "cirrus clouds",
    limit=100,
    output_dir="data/clouds/ci",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
)

# Repeat for other cloud types
downloader.download(
    "cumulus clouds",
    limit=100,
    output_dir="data/clouds/cu",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
)
