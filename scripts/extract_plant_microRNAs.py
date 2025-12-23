import sys, re, os, subprocess

def main():
    usage = 'Usage: ' + sys.argv[0] + ' <miRNA.dat.txt> <plant species>'

    if len(sys.argv) != 3:
        print(usage)
        sys.exit()

    miRNAdat_file = sys.argv[1]
    plant_species_file = sys.argv[2]

    plant_species = read_plant_species_file(plant_species_file)
    miRNAdat = read_miRNAdat_file(miRNAdat_file,plant_species)

###############
# SUBROUTINES # 
###############
    
def read_miRNAdat_file(miRNAdat_file,plant_species):
    miRNAdat = {}
    DATA1 = open('plant_mir_data.txt','w')
    DATA2 = open('plant_mir_cutsites.txt','w')
    drosha = 'Drosha'
    dicer = 'Dicer'
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
                    label = [0] * len(RNA)
                    L = len(RNA)
                    for span in spans:
                        start,end = span
                        label[start - 1] = 1
                        label[end - 1] = 1
                        if start < L/2:                        
                            cutsite_type = '5p'
                            print(f'{miR_name}\t{start}\t{cutsite_type+drosha}', file=DATA2)
                            print(f'{miR_name}\t{end}\t{cutsite_type+dicer}', file=DATA2)
                        else:
                            cutsite_type = '3p'
                            print(f'{miR_name}\t{start}\t{cutsite_type+dicer}', file=DATA2)
                            print(f'{miR_name}\t{end}\t{cutsite_type+drosha}', file=DATA2)
                    print(f'>{miR_name} {species} {spans}', file=DATA1)
                    print(''.join([str(x) for x in label]), file=DATA1)
                    print(f'{RNA}\n{dot_bracket}', file=DATA1)

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
