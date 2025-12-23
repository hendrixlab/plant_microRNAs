import sys, re, os, subprocess

def main():
    usage = 'Usage: ' + sys.argv[0] + ' <miRNA.dat.txt> <plant species>'

    if len(sys.argv) != 3:
        print(usage)
        sys.exit()

    miRNAdat_file = sys.argv[1]
    plant_species_file = sys.argv[2]

    plant_species = read_plant_species_file(plant_species_file)
    read_miRNAdat_file(miRNAdat_file,plant_species)

    
###############
# SUBROUTINES # 
###############
    
def read_miRNAdat_file(miRNAdat_file,plant_species):
    count = 0
    total = 0
    OFFSET = open('offsets.txt', 'w')
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
                    bp = get_base_pairs(dot_bracket)
                    L = len(RNA)
                    five_prime = None
                    three_prime = None
                    for span in spans:
                        start,end = span
                        if start < L/2:
                            five_prime = (start,end)
                        else:
                            three_prime = (start,end)
                    if five_prime and three_prime:
                        fstart,fend = five_prime
                        tstart,tend = three_prime
                        total += 1
                        if bp[fstart]:
                            count_fstart += 1
                            fstart_bp = bp[fstart]
                            dist = tend - fstart_bp
                            print(miR_name,'fstart',fstart,fstart_bp,tend,dist, file=OFFSET)
                        if bp[fend]:
                            count_fend += 1
                            fend_bp = bp[fend]
                            dist = tstart - fend_bp
                            print(miR_name,'fend',fend,fend_bp,tstart,dist, file=OFFSET)
                    elif five_prime and not three_prime:
                        fstart,fend = five_prime
                        fstart_bp = bp[fstart]
                        fend_bp = bp[fend]
                        # want tend = fstart_bp - 2
                        # want tstart = fend_bp - 2
                    elif not five_prime and three_prime:
                        tstart,tend = three_prime
                        tstart_bp = bp[tstart]
                        tend_bp = bp[tend]
                    else:
                        print('Error! Unexpected condition found.')
                        exit()
                    
                            
                            
    print(total, count_fstart, count_fstart/total, count_fend, count_fend/total)
                            
    return five_prime_starts,three_prime_starts
                        
def get_base_pairs(dot_bracket):
    bp = {}
    stack = []
    for i,x in enumerate(dot_bracket):
        if x == '(':
            stack.append(i)
        elif x == ')':
            j = stack.pop()
            # store in 1-based positions
            bp[i+1] = j+1
            bp[j+1] = i+1
        elif x == '.':
            # unpaired
            bp[i+1] = 0
    return bp

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
