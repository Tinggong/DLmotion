"""
Apply thresholds to each of the motion assessment measures in 'QAfrom-eddylog.txt' to select 
the motion-free volumes for usagage. 

A scheme file will be generated in the same subject folder, containing 1 for the selected image 
volumes and 0 for all other volumes. 

Usage: python3 SelectScheme.py --path SubjDir --t0 2 --t1 1.5 --r0 2 --r1 1.5 --outlier 0.05 --schemename scheme

If your diffusion datasets are collected with both AP and PA phase encoding direction at all the diffusion gradients,
and eddy_combine is used for the preprocessed diffusion dataset, add the following option to generate the scheme file:

python3 SelectScheme.py --path SubjDir --t0 2 --t1 1.5 --r0 2 --r1 1.5 --outlier 0.05 --schemename scheme --APPA True

Author: Zhiwei Li, Ting Gong
"""

import argparse

def load_eddy(path, thresholda=3, thresholdb=1.5, thresholdc=3, thresholdd=1.5, thresholde=0.05, schemename="scheme", APPA=False):
	"""
	"""
	filename = '/QAfrom-eddylog.txt'

	pfile = open(path + filename, 'r')

	lines = pfile.readlines()

	pfile.close()
	
	lineno = len(lines)

	print(lineno)
        
	move = []

	for i in range(lineno):
		line = lines[i]
		m = 0
		t0, t1, r0, r1, out = line.split(' ')
		t0, t1, r0, r1, out = float(t0), float(t1), float(r0), float(r1), float(out)
		if t0 < thresholda and t1 < thresholdb and r0 < thresholdc and r1 < thresholdd and out < thresholde:
			m = 1

		move.append(m)

	if APPA:
		mm = []
		AP = move[:int(lineno/2)]
		PA = move[int(lineno/2):]
		for i in range(int(lineno/2)):
			m=0
			if AP[i]==1 and PA[i]==1:
				m=1
			mm.append(m)
		move = mm

	move[0] = 1
	pfile = open(path + '/' + schemename, 'w')
	pfile.writelines("%s " % str(item) for item in move)
	pfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--t0", type=float)
    parser.add_argument("--t1", type=float)
    parser.add_argument("--r0", type=float)
    parser.add_argument("--r1", type=float)
    parser.add_argument("--outlier", type=float)
    parser.add_argument("--schemename")
    parser.add_argument("--APPA")
    args = parser.parse_args()

    load_eddy(args.path, args.t0, args.t1, args.r0, args.r1, args.outlier, args.schemename, args.APPA)
