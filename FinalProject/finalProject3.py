from ase.io import read, write
from ase.build import add_adsorbate
from ase import Atoms, Atom
from ase.build import molecule
from gpaw import GPAW, PW
from ase.optimize import BFGS
import numpy as np
import pickle
from ase.parallel import paropen

energies_compressive = {}
for system in ['BiCl2', 'BiS2', 'BiSCl']:
    for moleculas in ['H', 'OH', 'Li', 'Mg']:
        slab = read('relax--slab-'+system+'.traj') #equil to read('relax-BiClI.xyz.traj', -1)
        #print(slab)
        slab = slab.repeat((3, 3, 1))
        #print("supercell:", slab)
#         print('before:', slab.get_cell())
#         from ase.visualize import view
#         view(slab)        
        slab.set_cell(slab.get_cell()*[0.98, 0.98, 1],scale_atoms=True)  #add strain
#         print('after:', slab.get_cell())
#         from ase.visualize import view
#         view(slab)
#         assert False # Hack to break the execution by raising an error intentionally.
        top_site = slab[13].position
        bridge_site = (0.5 * (slab[13].x + slab[10].x), 0.5 * (slab[13].y + slab[10].y), slab[13].z)
        bridge_site = np.asarray(bridge_site) #transfer type of variable to array
        hollow_site = (1/3. * (slab[13].x + slab[10].x + slab[1].x), 1/3. * (slab[13].y + slab[10].y + slab[1].y), slab[13].z)
        hollow_site = np.asarray(hollow_site) #transfer type of variable to array
        for sites in ['top', 'bridge', 'hollow']:
            if sites == 'top':
                position = top_site
            elif sites == 'bridge':
                position = bridge_site
            elif sites == 'hollow':
                position = hollow_site
            
            name = system+'-'+moleculas+'-'+sites
            calc = GPAW(mode=PW(500),
                        xc='RPBE',
                        #kpts={'size': (5, 5, 1), 'gamma': True},
                        kpts={'density': 3, 'gamma': True},  
                        poissonsolver={'dipolelayer': 'xy'},
                        txt='out-'+name+'.out',
                        )
            
            ads = 0
            if moleculas == 'H':
                #add adsorbate
                ads = Atoms([Atom('H', (0., 0., 0.))])
                ads.translate(position + (0., 0., 1.5))  #Translate atomic positions, 13 -> top site                        
                
            if moleculas == 'OH':
                #add adsorbate
                ads = Atoms([Atom('O', (0., 0., 0.)),
                             Atom('H', (0., 0., 1.))])
                ads.translate(position + (0., 0., 1.5)) 
            
            if moleculas == 'Li':
                #add adsorbate
                ads = Atoms([Atom('Li', (0., 0., 0.))])
                ads.translate(position + (0., 0., 1.5)) 
            
            if moleculas == 'Mg':
                #add adsorbate
                ads = Atoms([Atom('Mg', (0., 0., 0.))])
                ads.translate(position + (0., 0., 1.5))

            slab_c = slab.copy()
            slab_c.extend(ads)
#             from ase.visualize import view
#             view(slab_c)
#             assert False # Hack to break the execution by raising an error intentionally.
            #relax
            slab_c.pbc = (True, True, False)
            slab_c.set_calculator(calc)
            relax_c = BFGS(slab_c, trajectory='relax--'+name+'.traj', logfile='log-relax-'+name+'.log')
            relax_c.run(fmax=0.1)
            energies_compressive[name] = slab_c.get_potential_energy()
            calc.write('gpw-relax-'+name+'.gpw')
        
        with paropen('energies_compressive.pckl', 'wb') as f:  #binary protocols output
            pickle.dump(energies_compressive, f)

#another side sites for BiSCl
slab = read('relax--slab-BiSCl.traj') 
slab = slab.repeat((3, 3, 1))
slab.set_cell(slab.get_cell()*[0.98, 0.98, 1],scale_atoms=True)  #add strain
top_back = slab[14].position
bridge_back = (0.5 * (slab[14].x + slab[26].x), 0.5 * (slab[14].y + slab[26].y), slab[14].z)
bridge_back = np.asarray(bridge_back) #transfer type of variable to array
hollow_back = (1/3. * (slab[14].x + slab[26].x + slab[17].x), 1/3. * (slab[14].y + slab[26].y + slab[17].y), slab[14].z)
hollow_back = np.asarray(hollow_back) #transfer type of variable to array
for moleculas in ['H', 'OH', 'Li', 'Mg']:
    for sites in ['top', 'bridge', 'hollow']:
        if sites == 'top':
            position = top_back
        elif sites == 'bridge':
            position = bridge_back
        elif sites == 'hollow':
            position = hollow_back
        
        name = 'BiSCl-'+moleculas+'-'+sites+'-back'
        calc = GPAW(mode=PW(500),
                xc='RPBE',
                #kpts={'size': (5, 5, 1), 'gamma': True},
                kpts={'density': 3, 'gamma': True}, 
                poissonsolver={'dipolelayer': 'xy'},
                txt='out-'+name+'.out',
                )
        
        ads = 0
        if moleculas == 'H':
            #add adsorbate
            ads = Atoms([Atom('H', (0., 0., 0.))])
            ads.translate(position + (0., 0., -1.5)) 
                
        elif moleculas == 'OH':
            #add adsorbate
            ads = Atoms([Atom('O', (0., 0., 0.)),
                         Atom('H', (0., 0., -1.))])
            ads.translate(position + (0., 0., -1.5))  
            
        elif moleculas == 'Li':
            #add adsorbate
            ads = Atoms([Atom('Li', (0., 0., 0.))])
            ads.translate(position + (0., 0., -1.5)) 
            
        elif moleculas == 'Mg':
            #add adsorbate
            ads = Atoms([Atom('Mg', (0., 0., 0.))])
            ads.translate(position + (0., 0., -1.5)) 
        
        slab_c = slab.copy()
        slab_c.extend(ads)
#         from ase.visualize import view
#         view(slab_c)
#         assert False # Hack to break the execution by raising an error intentionally.
        #relax
        slab_c.pbc = (True, True, False)
        slab_c.set_calculator(calc)
        relax_c = BFGS(slab_c, trajectory='relax--'+name+'.traj', logfile='log-relax-'+name+'.log')
        relax_c.run(fmax=0.1)
        energies_compressive[name] = slab_c.get_potential_energy()
        calc.write('gpw-relax-'+name+'.gpw')
    
    with paropen('energies_compressive.pckl', 'wb') as f:  #binary protocols output
        pickle.dump(energies_compressive, f)

#print(energies_compressive_slab)
with paropen('energies_compressive.pckl', 'wb') as f:  #binary protocols output
    pickle.dump(energies_compressive, f)
    
with paropen('energies_compressive.txt', 'w') as f:  # formatting output
    for x, y in energies_compressive.items():
        print('energy-{0:6}: {1:6.3f}'.format(x, y), file=f)


#         # Remember to visualize your structure
#         if 0: # for viewing only. Set to 1 to view, and 0 to ignore
#             from ase.visualize import view
#             #add_adsorbate(slab, 'H', 1.5, 'ontop')
#             view(slab)
#             assert False # Hack to break the execution by raising an error intentionally.

print('end............')
