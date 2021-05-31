from ase.io import read, write
from ase.build import add_adsorbate
from ase import Atoms, Atom
from ase.build import molecule
from gpaw import GPAW, PW
from ase.optimize import BFGS
import numpy as np
import pickle
from ase.parallel import paropen

tube = read('relax--slab-BiSCl_1D.traj')
#tube = read('relax--slab-BiS2.traj')
slab = tube.repeat((1, 1, 2))
#tube = read('BiSCl_1D.xyz')
# slab = tube.repeat((1, 1, 3))
from ase.visualize import view
view(slab)
assert False
# top_site = slab[83].position
# bridge_site = (0.5 * (slab[83].x + slab[131].x), 0.5 * (slab[83].y + slab[131].y), 0.5 * (slab[83].z + slab[131].z))
# bridge_site = np.asarray(bridge_site) #transfer type of variable to array
# hollow_site = (1/3. * (slab[83].x + slab[131].x + slab[132].x), 1/3. * (slab[83].y + slab[131].y + slab[132].y), 1/3. * (slab[83].z + slab[131].z + slab[132].z))
# hollow_site = np.asarray(hollow_site) #transfer type of variable to array
top_site = slab[35].position
bridge_site = (0.5 * (slab[35].x + slab[83].x), 0.5 * (slab[35].y + slab[83].y), 0.5 * (slab[35].z + slab[83].z))
bridge_site = np.asarray(bridge_site) #transfer type of variable to array
hollow_site = (1/3. * (slab[35].x + slab[83].x + slab[84].x), 1/3. * (slab[35].y + slab[83].y + slab[84].y), 1/3. * (slab[35].z + slab[83].z + slab[84].z))
hollow_site = np.asarray(hollow_site) #transfer type of variable to array
energies_tube = {}
#outer adsorption
for moleculas in ['H', 'OH', 'Li', 'Mg']:
    for sites in ['top', 'bridge', 'hollow']:
        if sites == 'top':
            position = top_site
        elif sites == 'bridge':
            position = bridge_site
        elif sites == 'hollow':
            position = hollow_site

        name = 'BiSCl_1D-'+moleculas+'-'+sites
        calc = GPAW(mode=PW(500),
                    xc='RPBE',
                    #kpts={'size': (5, 5, 1), 'gamma': True},
                    kpts={'density': 3, 'gamma': True},  
                    #poissonsolver={'dipolelayer': 'xy'},
                    txt='out-'+name+'.out',
                    )

        ads = 0
        if moleculas == 'H':
            #add adsorbate
            ads = Atoms([Atom('H', (0., 0., 0.))])
            ads.translate(position + (0., 1.5, 0.))  #Translate atomic positions, 13 -> top site                        

        if moleculas == 'OH':
            #add adsorbate
            ads = Atoms([Atom('O', (0., 0., 0.)),
                         Atom('H', (0., 1., 0.))])
            ads.translate(position + (0., 1.5, 0.)) 

        if moleculas == 'Li':
            #add adsorbate
            ads = Atoms([Atom('Li', (0., 0., 0.))])
            ads.translate(position + (0., 1.5, 0.)) 

        if moleculas == 'Mg':
            #add adsorbate
            ads = Atoms([Atom('Mg', (0., 0., 0.))])
            ads.translate(position + (0., 1.5, 0.))

        slab_c = slab.copy()
        slab_c.extend(ads)
#         from ase.visualize import view
#         view(slab_c)
#         assert False # Hack to break the execution by raising an error intentionally.
        #relax
        slab_c.pbc = (False, False, True)
        slab_c.set_calculator(calc)
        relax_c = BFGS(slab_c, trajectory='relax--'+name+'.traj', logfile='log-relax-'+name+'.log')
        relax_c.run(fmax=0.1)
        energies_tube[name] = slab_c.get_potential_energy()
        calc.write('gpw-relax-'+name+'.gpw')

    with paropen('energies_tube.pckl', 'wb') as f:  #binary protocols output
        pickle.dump(energies_tube, f)


#inner adsorption
# top_back = slab[68].position
# bridge_back = (0.5 * (slab[68].x + slab[116].x), 0.5 * (slab[68].y + slab[116].y), 0.5 * (slab[68].z + slab[116].z))
# bridge_back = np.asarray(bridge_back) #transfer type of variable to array
# hollow_back = (1/3. * (slab[68].x + slab[116].x + slab[67].x), 1/3. * (slab[68].y + slab[116].y + slab[67].y), 1/3. * (slab[68].z + slab[116].z + slab[67].z))
# hollow_back = np.asarray(hollow_back) #transfer type of variable to array
top_back = slab[20].position
bridge_back = (0.5 * (slab[20].x + slab[68].x), 0.5 * (slab[20].y + slab[68].y), 0.5 * (slab[20].z + slab[68].z))
bridge_back = np.asarray(bridge_back) #transfer type of variable to array
hollow_back = (1/3. * (slab[20].x + slab[68].x + slab[19].x), 1/3. * (slab[20].y + slab[68].y + slab[19].y), 1/3. * (slab[20].z + slab[68].z + slab[19].z))
hollow_back = np.asarray(hollow_back) #transfer type of variable to array
for moleculas in ['H', 'OH', 'Li', 'Mg']:
    for sites in ['top', 'bridge', 'hollow']:
        if sites == 'top':
            position = top_back
        elif sites == 'bridge':
            position = bridge_back
        elif sites == 'hollow':
            position = hollow_back
        
        name = 'BiSCl_1D-'+moleculas+'-'+sites+'-inner'
        calc = GPAW(mode=PW(500),
                xc='RPBE',
                #kpts={'size': (5, 5, 1), 'gamma': True},
                kpts={'density': 3, 'gamma': True}, 
                #poissonsolver={'dipolelayer': 'xy'},
                txt='out-'+name+'.out',
                )
        
        ads = 0
        if moleculas == 'H':
            #add adsorbate
            ads = Atoms([Atom('H', (0., 0., 0.))])
            ads.translate(position + (0., -1.5, 0.))  #Translate atomic positions, 13 -> top site                        

        if moleculas == 'OH':
            #add adsorbate
            ads = Atoms([Atom('O', (0., 0., 0.)),
                         Atom('H', (0., -1., 0.))])
            ads.translate(position + (0., -1.5, 0.)) 

        if moleculas == 'Li':
            #add adsorbate
            ads = Atoms([Atom('Li', (0., 0., 0.))])
            ads.translate(position + (0., -1.5, 0.)) 

        if moleculas == 'Mg':
            #add adsorbate
            ads = Atoms([Atom('Mg', (0., 0., 0.))])
            ads.translate(position + (0., -1.5, 0.))
        
        slab_c = slab.copy()
        slab_c.extend(ads)
#         from ase.visualize import view
#         view(slab_c)
#         assert False # Hack to break the execution by raising an error intentionally.
        #relax
        slab_c.pbc = (False, False, True)
        slab_c.set_calculator(calc)
        relax_c = BFGS(slab_c, trajectory='relax--'+name+'.traj', logfile='log-relax-'+name+'.log')
        relax_c.run(fmax=0.1)
        energies_tube[name] = slab_c.get_potential_energy()
        calc.write('gpw-relax-'+name+'.gpw')

    with paropen('energies_tube.pckl', 'wb') as f:  #binary protocols output
        pickle.dump(energies_tube, f)

#print(energies_tube)
with paropen('energies_tube.pckl', 'wb') as f:  #binary protocols output
    pickle.dump(energies_tube, f)
    
with paropen('energies_tube.txt', 'w') as f:  # formatting output
    for x, y in energies_tube.items():
        print('energy-{0:6}: {1:6.3f}'.format(x, y), file=f)


print('end............')
