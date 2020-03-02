<<CMT
strings=(
    bn
    or
    ur
)
for i in "${strings[@]}"; do
    find ./data -type f -name "*.alg.txt" -exec mv {} ./data/aligned \;
done
CMT
strings=(
    ta
    te
    ml
    mr
    hi
    gu
    or
    ur
)


for i in "${strings[@]}"; do
    mv ./mkb/pib.en-$i.* ./mkb/en-$i/
done

<<CMT
sorted=($(sort <<<"${strings[*]}"))


for i in "${sorted[@]}"; do
    mkdir -p ./mkb/$i-en/
    cat ./data/aligned/*.$i-en.$i.* > ./mkb/$i-en/mkb.$i
    cat ./data/aligned/*.$i-en.en.* > ./mkb/$i-en/mkb.en
done
CMT