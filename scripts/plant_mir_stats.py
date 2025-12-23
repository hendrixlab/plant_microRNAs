import sys, re, os, subprocess

def main():
    usage = 'Usage: ' + sys.argv[0] + ' <miRNA.dat.txt> <plant species>'

    if len(sys.argv) != 3:
        print(usage)
        sys.exit()

    miRNAdat_file = sys.argv[1]
    plant_species_file = sys.argv[2]

    plant_species = read_plant_species_file(plant_species_file)
    five_prime_starts,three_prime_starts = read_miRNAdat_file(miRNAdat_file,plant_species)

    five_prime_L = list(five_prime_starts.keys())
    three_prime_L = list(three_prime_starts.keys())

    five_prime_L.sort()
    three_prime_L.sort()
    
    for L in five_prime_L:
        print('5p',L,len(five_prime_starts[L]),min(five_prime_starts[L]),max(five_prime_starts[L]),max(five_prime_starts[L])-min(five_prime_starts[L]))
        
    for L in three_prime_L:
        print('3p',L,len(three_prime_starts[L]),min(three_prime_starts[L]),max(three_prime_starts[L]),max(three_prime_starts[L])-min(three_prime_starts[L]))
    
###############
# SUBROUTINES # 
###############
    
def read_miRNAdat_file(miRNAdat_file,plant_species):
    five_prime_starts = {}
    three_prime_starts = {}    
    with open(miRNAdat_file) as F:
        for line in F:
            if line.startswith('ID'):
                # store previous data
                # beginning of record, initialize
                terms = line.strip().split()
                miR_name = terms[1]
                spans = []
                pre_miR = ''
                species = ''
            if line.startswith('DE'):
                # get species/micRNA ID
                terms = line.strip().split()
                species = ' '.join(terms[1:1+2])
            if line.startswith('FT'):
                # store microRNA positions
                terms = line.strip().split()
                if terms[1] == 'miRNA':
                    if len(terms) != 3:
                        print('error')
                        exit()
                    else:
                        pos = terms[2]
                        start,end = pos.split('..')
                        span = (int(start),int(end))
                        spans.append(span)
            if line.startswith('SQ'):
                # can store current SQ line info
                line = next(F) # first sequence data
                while not line.startswith('//'):
                    terms = line.strip().split()
                    if not is_int(terms[-1]):
                        print('no int found for SQ block')
                        exit()
                    else:
                        seq_part = ''.join(terms[0:-1])
                        pre_miR += seq_part
                    line = next(F)
                if species in plant_species:
                    RNA = pre_miR.upper()
                    dot_bracket,mfe = run_RNAfold(RNA)
                    L = len(RNA)
                    for span in spans:
                        start,end = span
                        if start < L/2:
                            if L not in five_prime_starts:
                                five_prime_starts[L] = []
                            five_prime_starts[L].append(start)
                        else:
                            if L not in three_prime_starts:
                                three_prime_starts[L] = []
                            three_prime_starts[L].append(start)
    return five_prime_starts,three_prime_starts
                        

def read_plant_species_file(plant_species_file):
    plant_species = {}
    with open(plant_species_file) as F:
        for line in F:
            plant_name = line.strip()
            plant_species[plant_name] = True
    return plant_species

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def run_RNAfold(RNA):
    output = subprocess.check_output("echo " + str(RNA) + " | RNAfold --noPS", shell=True, universal_newlines=True)
    lines = output.split('\n')
    #print(RNA, lines)
    terms = lines[1].split(' ')
    dot_bracket = terms[0]
    mfe_info = ''.join(terms[1:])
    mfe = re.sub(r'[()]','',mfe_info)
    #print(lines[1],mfe)
    return dot_bracket,float(mfe)


    
# main
main()
