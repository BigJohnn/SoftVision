#!/bin/bash

echo ${pwd}

x=`find . -name '*.metal'`
echo ${x}
for i in ${x}
do
	shaderfilename=${i%.*}
	echo ===$shaderfilename
	xcrun -sdk iphoneos metal -c ${i} -o ${shaderfilename}.air
done

y=`find . -name '*.air'`
echo $y
xcrun -sdk iphoneos metallib ${y} -o sgm.metallib

xcrun -sdk iphoneos metal -frecord-sources -o sgm.metallib ${y}

xcrun -sdk iphoneos metal-dsymutil -flat -remove-source sgm.metallib

echo 'done!!'