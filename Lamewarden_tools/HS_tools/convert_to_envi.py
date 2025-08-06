import sys
def convert_header_to_envi(bil_header):
    # Open the BIL header file and read its lines
    with open(bil_header, 'r') as f:
        lines = f.readlines()
        # Save the original BIL header file with a new name
        original_header_path = f'{bil_header}_original'
        with open(original_header_path, 'w') as f:
            f.writelines(lines)

    # Initialize a dictionary to hold the header information
    header_info = {}
    wavelengths = []

    # Parse the BIL header file
    in_wavelengths = False
    for line in lines:
        if line.strip():
            if 'WAVELENGTHS' in line:
                in_wavelengths = True
            elif 'WAVELENGTHS_END' in line:
                in_wavelengths = False
            elif in_wavelengths:
                wavelengths.append(line.strip())
            else:
                key, value = line.split()
                header_info[key] = value

    # Write the ENVI header file
    with open(bil_header, 'w') as f:
        f.write('ENVI\n')
        f.write('file type = ENVI\n')
        f.write(f'interleave = {header_info["LAYOUT"]}\n')
        f.write(f'samples = {header_info["NCOLS"]}\n')
        f.write(f'lines = {header_info["NROWS"]}\n')
        f.write(f'bands = {header_info["NBANDS"]}\n')
        # selecting the data type
        if header_info["NBITS"] == '16':
            f.write(f'data type = 2\n')
        elif header_info["NBITS"] == '32':
            f.write(f'data type = 3\n')
        elif header_info["NBITS"] == '64':
            f.write(f'data type = 14\n')
        elif header_info["NBITS"] == '12':
            f.write(f'data type = 12\n')
            # 64-bit unsigned int
        if header_info["BYTEORDER"] == 'I':
            f.write(f'byte order = 0\n')
        else:
            f.write(f'byte order = 1\n')
            f.write(f'byte order = {header_info["BYTEORDER"]}\n')
            
        f.write('wavelength units = nm\n')
        f.write(f';bit depth = {header_info["NBITS"]}\n')
        f.write(f';chromatic correction = {header_info["CHROMATICCORRECTION"]}\n')
        f.write(f';integration time = {header_info["INTEGRATIONTIME"]}\n')
        try:
            f.write(f';gain = {header_info["GAIN"]}\n')
        except:
            f.write(f';gain = 0\n')
        f.write('wavelength = {\n')
        for wavelength in wavelengths:
            f.write(f'{wavelength},\n')
        f.write('}\n')

# Check if the script is being run as standalone
if __name__ == "__main__":
    # Check if the required argument is provided
    if len(sys.argv) < 2:
        print("Please provide the path to the BIL header file as an argument.")
    else:
        # Get the path to the BIL header file from the command line argument
        bil_header_path = sys.argv[1]
        
        # Use the function
        convert_header_to_envi(bil_header_path)
