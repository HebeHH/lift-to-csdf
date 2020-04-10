
# nbconvert
for f in *.ipynb
do
	echo "'"$f"'"
	fpy="${f%%.*}"".py"
	echo $fpy
	# nbconvert
	jupyter nbconvert --to script $f
	# change %run .ipynb to import .py 
	sed -i -E "s/^get_ipython().run_line_magic('run', '\([a-zA-Z]*\).ipynb')/import \1/" $fpy
	# move file
	mv $fpy src/
done
rm *.py-E



# change %run .ipynb to import .py
# sed -i -E "s/^get_ipython().run_line_magic('run', '\([a-zA-Z]*\).ipynb')/import \1/" test.txt