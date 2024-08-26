#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=8:00:00
#SBATCH --job-name=save_cats
#SBATCH -p gpu
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.out
#SBATCH --error=/mnt/home/spandey/ceph/CHARM/run_scripts/slurm_logs/%x.%j.err

module purge
module load python
module load cuda
module load cudnn
module load nccl
source ~/miniconda3/bin/activate ili-sbi

cd /mnt/home/spandey/ceph/CHARM/charm/


srun --time=1 python predict_save_halo_cats.py 1800
srun --time=1 python predict_save_halo_cats.py 1801
srun --time=1 python predict_save_halo_cats.py 1802
srun --time=1 python predict_save_halo_cats.py 1803
srun --time=1 python predict_save_halo_cats.py 1804
srun --time=1 python predict_save_halo_cats.py 1805
srun --time=1 python predict_save_halo_cats.py 1806
srun --time=1 python predict_save_halo_cats.py 1807
srun --time=1 python predict_save_halo_cats.py 1808
srun --time=1 python predict_save_halo_cats.py 1809
srun --time=1 python predict_save_halo_cats.py 1810
srun --time=1 python predict_save_halo_cats.py 1811
srun --time=1 python predict_save_halo_cats.py 1812
srun --time=1 python predict_save_halo_cats.py 1813
srun --time=1 python predict_save_halo_cats.py 1814
srun --time=1 python predict_save_halo_cats.py 1815
srun --time=1 python predict_save_halo_cats.py 1816
srun --time=1 python predict_save_halo_cats.py 1817
srun --time=1 python predict_save_halo_cats.py 1818
srun --time=1 python predict_save_halo_cats.py 1819
srun --time=1 python predict_save_halo_cats.py 1820
srun --time=1 python predict_save_halo_cats.py 1821
srun --time=1 python predict_save_halo_cats.py 1822
srun --time=1 python predict_save_halo_cats.py 1823
srun --time=1 python predict_save_halo_cats.py 1824
srun --time=1 python predict_save_halo_cats.py 1825
srun --time=1 python predict_save_halo_cats.py 1826
srun --time=1 python predict_save_halo_cats.py 1827
srun --time=1 python predict_save_halo_cats.py 1828
srun --time=1 python predict_save_halo_cats.py 1829
srun --time=1 python predict_save_halo_cats.py 1830
srun --time=1 python predict_save_halo_cats.py 1831
srun --time=1 python predict_save_halo_cats.py 1832
srun --time=1 python predict_save_halo_cats.py 1833
srun --time=1 python predict_save_halo_cats.py 1834
srun --time=1 python predict_save_halo_cats.py 1835
srun --time=1 python predict_save_halo_cats.py 1836
srun --time=1 python predict_save_halo_cats.py 1837
srun --time=1 python predict_save_halo_cats.py 1838
srun --time=1 python predict_save_halo_cats.py 1839
srun --time=1 python predict_save_halo_cats.py 1840
srun --time=1 python predict_save_halo_cats.py 1841
srun --time=1 python predict_save_halo_cats.py 1842
srun --time=1 python predict_save_halo_cats.py 1843
srun --time=1 python predict_save_halo_cats.py 1844
srun --time=1 python predict_save_halo_cats.py 1845
srun --time=1 python predict_save_halo_cats.py 1846
srun --time=1 python predict_save_halo_cats.py 1847
srun --time=1 python predict_save_halo_cats.py 1848
srun --time=1 python predict_save_halo_cats.py 1849
srun --time=1 python predict_save_halo_cats.py 1850
srun --time=1 python predict_save_halo_cats.py 1851
srun --time=1 python predict_save_halo_cats.py 1852
srun --time=1 python predict_save_halo_cats.py 1853
srun --time=1 python predict_save_halo_cats.py 1854
srun --time=1 python predict_save_halo_cats.py 1855
srun --time=1 python predict_save_halo_cats.py 1856
srun --time=1 python predict_save_halo_cats.py 1857
srun --time=1 python predict_save_halo_cats.py 1858
srun --time=1 python predict_save_halo_cats.py 1859
srun --time=1 python predict_save_halo_cats.py 1860
srun --time=1 python predict_save_halo_cats.py 1861
srun --time=1 python predict_save_halo_cats.py 1862
srun --time=1 python predict_save_halo_cats.py 1863
srun --time=1 python predict_save_halo_cats.py 1864
srun --time=1 python predict_save_halo_cats.py 1865
srun --time=1 python predict_save_halo_cats.py 1866
srun --time=1 python predict_save_halo_cats.py 1867
srun --time=1 python predict_save_halo_cats.py 1868
srun --time=1 python predict_save_halo_cats.py 1869
srun --time=1 python predict_save_halo_cats.py 1870
srun --time=1 python predict_save_halo_cats.py 1871
srun --time=1 python predict_save_halo_cats.py 1872
srun --time=1 python predict_save_halo_cats.py 1873
srun --time=1 python predict_save_halo_cats.py 1874
srun --time=1 python predict_save_halo_cats.py 1875
srun --time=1 python predict_save_halo_cats.py 1876
srun --time=1 python predict_save_halo_cats.py 1877
srun --time=1 python predict_save_halo_cats.py 1878
srun --time=1 python predict_save_halo_cats.py 1879
srun --time=1 python predict_save_halo_cats.py 1880
srun --time=1 python predict_save_halo_cats.py 1881
srun --time=1 python predict_save_halo_cats.py 1882
srun --time=1 python predict_save_halo_cats.py 1883
srun --time=1 python predict_save_halo_cats.py 1884
srun --time=1 python predict_save_halo_cats.py 1885
srun --time=1 python predict_save_halo_cats.py 1886
srun --time=1 python predict_save_halo_cats.py 1887
srun --time=1 python predict_save_halo_cats.py 1888
srun --time=1 python predict_save_halo_cats.py 1889
srun --time=1 python predict_save_halo_cats.py 1890
srun --time=1 python predict_save_halo_cats.py 1891
srun --time=1 python predict_save_halo_cats.py 1892
srun --time=1 python predict_save_halo_cats.py 1893
srun --time=1 python predict_save_halo_cats.py 1894
srun --time=1 python predict_save_halo_cats.py 1895
srun --time=1 python predict_save_halo_cats.py 1896
srun --time=1 python predict_save_halo_cats.py 1897
srun --time=1 python predict_save_halo_cats.py 1898
srun --time=1 python predict_save_halo_cats.py 1899
srun --time=1 python predict_save_halo_cats.py 1900
srun --time=1 python predict_save_halo_cats.py 1901
srun --time=1 python predict_save_halo_cats.py 1902
srun --time=1 python predict_save_halo_cats.py 1903
srun --time=1 python predict_save_halo_cats.py 1904
srun --time=1 python predict_save_halo_cats.py 1905
srun --time=1 python predict_save_halo_cats.py 1906
srun --time=1 python predict_save_halo_cats.py 1907
srun --time=1 python predict_save_halo_cats.py 1908
srun --time=1 python predict_save_halo_cats.py 1909
srun --time=1 python predict_save_halo_cats.py 1910
srun --time=1 python predict_save_halo_cats.py 1911
srun --time=1 python predict_save_halo_cats.py 1912
srun --time=1 python predict_save_halo_cats.py 1913
srun --time=1 python predict_save_halo_cats.py 1914
srun --time=1 python predict_save_halo_cats.py 1915
srun --time=1 python predict_save_halo_cats.py 1916
srun --time=1 python predict_save_halo_cats.py 1917
srun --time=1 python predict_save_halo_cats.py 1918
srun --time=1 python predict_save_halo_cats.py 1919
srun --time=1 python predict_save_halo_cats.py 1920
srun --time=1 python predict_save_halo_cats.py 1921
srun --time=1 python predict_save_halo_cats.py 1922
srun --time=1 python predict_save_halo_cats.py 1923
srun --time=1 python predict_save_halo_cats.py 1924
srun --time=1 python predict_save_halo_cats.py 1925
srun --time=1 python predict_save_halo_cats.py 1926
srun --time=1 python predict_save_halo_cats.py 1927
srun --time=1 python predict_save_halo_cats.py 1928
srun --time=1 python predict_save_halo_cats.py 1929
srun --time=1 python predict_save_halo_cats.py 1930
srun --time=1 python predict_save_halo_cats.py 1931
srun --time=1 python predict_save_halo_cats.py 1932
srun --time=1 python predict_save_halo_cats.py 1933
srun --time=1 python predict_save_halo_cats.py 1934
srun --time=1 python predict_save_halo_cats.py 1935
srun --time=1 python predict_save_halo_cats.py 1936
srun --time=1 python predict_save_halo_cats.py 1937
srun --time=1 python predict_save_halo_cats.py 1938
srun --time=1 python predict_save_halo_cats.py 1939
srun --time=1 python predict_save_halo_cats.py 1940
srun --time=1 python predict_save_halo_cats.py 1941
srun --time=1 python predict_save_halo_cats.py 1942
srun --time=1 python predict_save_halo_cats.py 1943
srun --time=1 python predict_save_halo_cats.py 1944
srun --time=1 python predict_save_halo_cats.py 1945
srun --time=1 python predict_save_halo_cats.py 1946
srun --time=1 python predict_save_halo_cats.py 1947
srun --time=1 python predict_save_halo_cats.py 1948
srun --time=1 python predict_save_halo_cats.py 1949
srun --time=1 python predict_save_halo_cats.py 1950
srun --time=1 python predict_save_halo_cats.py 1951
srun --time=1 python predict_save_halo_cats.py 1952
srun --time=1 python predict_save_halo_cats.py 1953
srun --time=1 python predict_save_halo_cats.py 1954
srun --time=1 python predict_save_halo_cats.py 1955
srun --time=1 python predict_save_halo_cats.py 1956
srun --time=1 python predict_save_halo_cats.py 1957
srun --time=1 python predict_save_halo_cats.py 1958
srun --time=1 python predict_save_halo_cats.py 1959
srun --time=1 python predict_save_halo_cats.py 1960
srun --time=1 python predict_save_halo_cats.py 1961
srun --time=1 python predict_save_halo_cats.py 1962
srun --time=1 python predict_save_halo_cats.py 1963
srun --time=1 python predict_save_halo_cats.py 1964
srun --time=1 python predict_save_halo_cats.py 1965
srun --time=1 python predict_save_halo_cats.py 1966
srun --time=1 python predict_save_halo_cats.py 1967
srun --time=1 python predict_save_halo_cats.py 1968
srun --time=1 python predict_save_halo_cats.py 1969
srun --time=1 python predict_save_halo_cats.py 1970
srun --time=1 python predict_save_halo_cats.py 1971
srun --time=1 python predict_save_halo_cats.py 1972
srun --time=1 python predict_save_halo_cats.py 1973
srun --time=1 python predict_save_halo_cats.py 1974
srun --time=1 python predict_save_halo_cats.py 1975
srun --time=1 python predict_save_halo_cats.py 1976
srun --time=1 python predict_save_halo_cats.py 1977
srun --time=1 python predict_save_halo_cats.py 1978
srun --time=1 python predict_save_halo_cats.py 1979
srun --time=1 python predict_save_halo_cats.py 1980
srun --time=1 python predict_save_halo_cats.py 1981
srun --time=1 python predict_save_halo_cats.py 1982
srun --time=1 python predict_save_halo_cats.py 1983
srun --time=1 python predict_save_halo_cats.py 1984
srun --time=1 python predict_save_halo_cats.py 1985
srun --time=1 python predict_save_halo_cats.py 1986
srun --time=1 python predict_save_halo_cats.py 1987
srun --time=1 python predict_save_halo_cats.py 1988
srun --time=1 python predict_save_halo_cats.py 1989
srun --time=1 python predict_save_halo_cats.py 1990
srun --time=1 python predict_save_halo_cats.py 1991
srun --time=1 python predict_save_halo_cats.py 1992
srun --time=1 python predict_save_halo_cats.py 1993
srun --time=1 python predict_save_halo_cats.py 1994
srun --time=1 python predict_save_halo_cats.py 1995
srun --time=1 python predict_save_halo_cats.py 1996
srun --time=1 python predict_save_halo_cats.py 1997
srun --time=1 python predict_save_halo_cats.py 1998
srun --time=1 python predict_save_halo_cats.py 1999


echo "All done!"
