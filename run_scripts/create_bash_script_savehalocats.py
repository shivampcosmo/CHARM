
    
ldir = '/mnt/home/spandey/ceph/CHARM/run_scripts'
ffile_orig = open(ldir + '/submit_save_halocats_template.sh', 'r')
f = ffile_orig.readlines()
ffile_orig.close()


g = ''   
for line in f:
    g += line
            

g += '\n\n'
n1 = 1800
n2 = 2000
for ji in range(n1, n2):
    g += f'srun --time=1 python predict_save_halo_cats.py {ji}\n'

g += '\n\n'
g += 'echo "All done!"\n'

# print(g)
ffile = open(ldir + f'/submit_save_halocats_{n1}_{n2}.sh', 'w')
ffile.write(g)
ffile.close()
