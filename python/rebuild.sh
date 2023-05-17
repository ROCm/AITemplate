rm -rf /root/.aitemplate/rocm.db
rm -rf tmp
rm -rf build/
python3 setup.py bdist_wheel
pip install dist/*.whl --force-reinstall