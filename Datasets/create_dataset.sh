#!/usr/bin/env sh

# Ensure directories are present
if [ ! -d "2018 inkless+wallpaper/" ] || [ ! -d "2018 schuhe+spezial/" ] || [ ! -d "2019 inkless+wallpaper/" ] || [ ! -d "2019 schuhe+spezial/" ]; then
    echo "Missing files"
    echo "Please download dataset, extract all archives and then run this in the root of the dataset."
    exit
fi

if ! command -v convert >/dev/null 2>&1; then
    echo "Missing ImageMagick, please install"
    exit
fi

# Create dataset directories
mkdir -p "Dataset/Gallery"
mkdir -p "Dataset/Query"

num_cores=$(nproc)

# --------- Gallery Images ---------

echo "Copying gallery images"

find "2018 inkless+wallpaper/" -type f -name "*_1_R.jpg" -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")
    filename=$(echo "$filename" | sed 's/_[0-9]_[A-Za-z]*//')

    cp "$file" "Dataset/Gallery/$filename"
done

find "2019 inkless+wallpaper/" -type f -name "*_3_1.jpg" -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")
    filename=$(echo "$filename" | sed 's/_[0-9]_[0-9]//')

    cp "$file" "Dataset/Gallery/$filename"
done

echo "Copying query images"

# --------- Query Images ---------

find "2018 inkless+wallpaper/" -type f -regex '.*[0-9]+_[a-z]_[A-Z]\.jpg' -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")

    cp "$file" "Dataset/Query/$filename"
done

find "2019 inkless+wallpaper/" -type f -regex '.*[0-9]+_[4-5]_[0-9]\.jpg' -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")

    cp "$file" "Dataset/Query/$filename"
done

find "2018 schuhe+spezial/" -type f -regextype posix-egrep -regex '.*[^_][0-9]{3}_0[269]_.*' -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")

    if expr "$filename" : "^[0-9][0-9]9_.*" 1>/dev/null; then
        cp "$file" "Dataset/Query/$filename"
    else
        # Rotate 90 degrees
        eval "convert \"$file\" -rotate 90 \"Dataset/Query/$filename\"" &

        # Only use cores availible
        while [ "$(jobs | wc -l)" -ge "$num_cores" ]; do
            sleep 1
        done
    fi

done

find "2019 schuhe+spezial/" -type f -regextype posix-egrep \( -regex '.*[0-9]{3}_0[269]_.*' -o -regex '.*[0-9]{2}_[0-9]{4}.jpg' \) -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")

    # Paper prints (ending in 9) dont need rotated
    if expr "$filename" : "^[0-9]*9_.*" 1>/dev/null; then
        cp "$file" "Dataset/Query/$filename"
    elif expr "$filename" : "^[0-9]*_.*" 1>/dev/null; then
        # Rotate 90 degrees
        eval "convert \"$file\" -rotate 90 \"Dataset/Query/$filename\"" &

        # Only use cores availible
        while [ "$(jobs | wc -l)" -ge "$num_cores" ]; do
            sleep 1
        done
    fi

done

# --------- Edge Cases and Corrupted Files ---------

echo "Fixing edge cases and corrupted files"
mv "Dataset/Gallery/1.1.jpg" "Dataset/Gallery/1.jpg" # Duplicate gallery images
rm "Dataset/Query/92_5_2.jpg" "Dataset/Query/92_5_3.jpg" "Dataset/Query/135_4_2.jpg" # Corrupted images
rm "Dataset/Query/154_02_2.jpg" "Dataset/Query/154_02_3.jpg" # Incorrectly named shoe sole images

# Fix premature end of JPEG file
find "Dataset/Query/" -type f -regextype posix-egrep -regex '.*[0-9]+_[4-5]_[2-3].jpg' -print | while IFS= read -r file; do
    # Create new filename
    filename=$(basename "$file")

    eval "convert \"$file\" -strip \"Dataset/Query/$filename\" >/dev/null 2>&1" &

    # Only use cores availible
    while [ "$(jobs | wc -l)" -ge "$num_cores" ]; do
        sleep 1
    done

done


# convert "Dataset/Query/191_5_3.jpg" -strip "Dataset/Query/191_5_3.jpg" >/dev/null 2>&1
# convert "Dataset/Query/194_5_3.jpg" -strip "Dataset/Query/194_5_3.jpg" >/dev/null 2>&1
# convert "Dataset/Query/200_5_3.jpg" -strip "Dataset/Query/200_5_3.jpg" >/dev/null 2>&1
