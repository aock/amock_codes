import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import copy
import random
from pprint import pprint
from tqdm import tqdm


class GrowingNN:

    ACTIVATIONS = ['relu', 'linear', 'tanh', 'softmax', 'sigmoid']

    OPTIMIZERS = ['adam', 'rmsprop', 'mse', 'sgd']

    PARAMETER_LAYER_TYPES = ['dense']

    CONSTANT_LAYER_TYPES = ['dropout']


    def __init__(self,
                init_model = None,
                input_shape=(200,),
                num_output=1,
                max_population=10,
                n_childs_per_gen=10,
                p_mutate = 0.5,
                p_add_layer = 0.2,
                p_del_layer = 0.2):
        

        self.max_population = max_population
        self.n_childs_per_gen = n_childs_per_gen
        self.p_mutate = p_mutate
        self.p_add_layer = p_add_layer
        self.p_del_layer = p_del_layer

        if init_model is None:
            self.model_descr = {
                'net': [
                    {
                        'type':'input',
                        'params': {
                            'shape':input_shape
                        }
                    },
                    {
                        'type':'dense',
                        'params': {
                            'units':32,
                            'activation': 'relu'
                        }
                    },
                    {
                        'type':'dropout',
                        'params': {
                            'rate':0.2
                        }
                    },
                    {
                        'type':'dense',
                        'params':{
                            'units':11,
                            'activation':'relu'
                        }
                    },
                    {
                        'type':'dropout',
                        'params':{
                            'rate':0.2
                        }
                    },
                    {
                        'type':'dense',
                        'params':{
                            'units':22,
                            'activation':'relu'
                        }
                    },
                    {
                        'type':'dropout',
                        'params':{
                            'rate':0.3
                        }
                    },
                    {
                        'type':'dense',
                        'params': {
                            'units':num_output,
                            'activation':'relu'
                        }
                    }
                ],
                'compile': {
                    'loss':'binary_crossentropy',
                    'optimizer':'adam',
                    'metrics':['accuracy']
                }
            }
        else:
            self.model_descr = init_model

        # Model population:
        # entry: [model description, generation, gen_id, fitness]
        self.model_population = []
        self.model_population.append([self.model_descr, 0, 0, 0.0, 0.0])
        self.current_gen = 0


    def __build_model(self, descr):
        
        net = descr['net']

        input_layer = self.__keras_from_descr(net[0])
        x = input_layer
        for i in range(len(net)-1):

            keras_elem = self.__keras_from_descr(net[i+1])

            x = keras_elem(x)

        model = Model(inputs=input_layer, outputs=x)

        compile_params = descr['compile']
        model.compile(loss=compile_params['loss'],
                      optimizer=compile_params['optimizer'],
                      metrics=compile_params['metrics']
                      )

        return model

    def __keras_from_descr(self, descr):
        if descr['type'] in ['input','Input']:
            return Input(**descr['params'])
        elif descr['type'] in ['dense','Dense']:
            return Dense(**descr['params'])
        elif descr['type'] in ['dropout','Dropout']:
            return Dropout(**descr['params'])
        else:
            raise TypeError('Error: Type ' + descr['type'] + ' not supported')

    def __compute_fitness(self, losses, accs, val_losses, val_accs):

        # when is a model better ?
        # last accuracy is important
        last_loss = losses[-1]
        last_acc = accs[-1]
        last_val_loss = val_losses[-1]
        last_val_acc = val_accs[-1]

        scores = []
        
        # Compare last_acc with last_loss
        # if last_acc is low and last_loss is low -> underfitting
        # if last_acc is high and last_loss is low -> good
        # if last_acc is high and last_loss is high -> good
        # if last_acc is low and last_loss is high -> potential good, more training (local min)

        # functon -> high good, low bad
        add_pot = [0, 0]
        sum_pot = sum(add_pot)
        cmp_acc_loss = ( last_val_acc**(add_pot[0]+1) / last_val_loss**(add_pot[1]+1) )**(1/(sum_pot+1) )
        # print(cmp_acc_loss)
        scores.append(cmp_acc_loss)

        # Compare validation and training
        # if last_acc is low and last_val_acc is low -> underfitting
        # if last_acc is high and last_val_acc is low -> overfitting
        # if last_acc is high and last_val_acc is high -> good
        # if last_acc is low and last_val_acc is high -> potential good (randomly found)

        # **4 -> if low last_val_acc -> more important for low value
        add_pot = [0, 3]
        sum_pot = sum(add_pot)
        cmp_acc_val_acc = (last_acc**(add_pot[0]+1) * last_val_acc**(add_pot[1]+1) )**(1/(sum_pot+1))
        # print(cmp_acc_val_acc)
        scores.append(cmp_acc_val_acc)

        # if last_loss is low and last_val_loss is low -> good
        # if last_loss is high and last_val_loss is low -> potential good
        # if last_loss is high and last_val_loss is high -> underfitting
        # if last_loss is low and last_val_loss is high -> overfitting

        add_pot = [0, 3]
        sum_pot = sum(add_pot)
        cmp_loss_val_loss = ( (1/last_loss)**(add_pot[0]+1) * (1/last_val_loss)**(add_pot[1]+1) )**(1/(sum_pot+1) )
        scores.append(cmp_loss_val_loss)

        scores = np.array(scores)
        score = np.prod(scores)

        return score

    def __update_fitnesses(self, X, y, pbar = None):

        for i in range(len(self.model_population)):
            elem = self.model_population[i]

            # only fit teenagers
            if elem[1] == self.current_gen - 1:
                model = self.__build_model(elem[0])

                # train model
                hist = model.fit(X, y, epochs=500, callbacks=[EarlyStopping(monitor='val_loss', patience=15)], shuffle=True, validation_split=0.2, verbose=0)
                losses = hist.history['loss']
                accs = hist.history['acc']
                val_losses = hist.history['val_loss']
                val_accs = hist.history['val_acc']

                # compute fitness
                fitness = self.__compute_fitness(losses, accs, val_losses, val_accs)

                # update fitness
                self.model_population[i][3] = fitness
                self.model_population[i][4] = val_accs[-1]

            if not pbar is None:
                pbar.update(1)

    def __rank_population(self):
        self.model_population.sort(key=lambda x: x[3], reverse=True)

    def __die_hard(self):

        number_of_deaths = len(self.model_population) - self.max_population

        if number_of_deaths > 0:
            
            for i in range(number_of_deaths):
                del self.model_population[-1]

            return number_of_deaths
            
        else:
            return 0

    def __gen_child(self, modelA, modelB):
        netA = modelA[0]['net']
        netB = modelB[0]['net']

        num_dynamic_layers_A = len(netA)-2
        num_dyn_param_layer_A = int(num_dynamic_layers_A / 2)

        num_dynamic_layers_B = len(netB)-2
        num_dyn_param_layer_B = int(num_dynamic_layers_B / 2)

        n_max_layers = np.max([num_dyn_param_layer_A, num_dyn_param_layer_B])

        childNet = [netA[0]]

        fits = np.array([modelA[3], modelB[3]])
        probs = fits / np.sum(np.abs(fits))

        for i in range(n_max_layers):
            idx = i*2+1
            if idx > num_dynamic_layers_A:
                # out of bounds for netA
                choices = [ None, netB[idx:idx+2] ]
                choice_ids = [0, 1]
                elem_id = np.random.choice(choice_ids, p=probs)
                elems = choices[elem_id]
                if not elems is None:
                    childNet.extend(elems)
                
            elif idx > num_dynamic_layers_B:
                # out of bounds for netB
                choices = [ netA[idx:idx+2],None ]
                choice_ids = [0, 1]
                elem_id = np.random.choice(choice_ids, p=probs)
                elems = choices[elem_id]
                if not elems is None:
                    childNet.extend(elems)
            else:
                # not out of bounds for both netA and netB
                choices = [ netA[idx:idx+2],netB[idx:idx+2] ]
                choice_ids = [0, 1]
                elem_id = np.random.choice(choice_ids, p=probs)
                elems = choices[elem_id]
                childNet.extend(elems)

        childNet.append(netA[-1])

        choices = [modelA, modelB]
        choice_ids = [0, 1]
        choice_id = np.random.choice(choice_ids, p=probs)
        child = copy.deepcopy(choices[choice_id])
        child[3] = 0.0
        child[4] = 0.0
        child[0]['net'] = childNet
        return child

    def __gen_childs(self, population, n_first_add = 3, n_childs = 5):
        
        childs = []

        if len(population) == 1:
            # generate a n_first_add random instances
            
            for i in range(n_first_add):
                new_descr = copy.deepcopy(population[0][0])
                old_net = new_descr['net']

                new_net = [old_net[0]]
                
                for j in range(len(old_net)-2):
                    elem = old_net[j+1]
                    new_elem = self.__random_mutate_elem(elem, p=0.5)
                    new_net.append(new_elem)

                new_net.append(old_net[-1])
                new_descr['net'] = new_net

                childs.append([new_descr, self.current_gen, i, 0.0, 0.0])

        else:
            # generate childs

            fittnesses = []
            for elem in population:
                fittnesses.append(elem[3])

            fittnesses = np.array(fittnesses)
            # normalized to probabilities
            probs = fittnesses / np.sum(np.abs(fittnesses))
            indices = np.arange(0, len(population))

            for i in range(n_childs):

                father_id = np.random.choice(indices, p=probs)
                mother_id = np.random.choice(indices, p=probs)
                father = population[father_id]
                mother = population[mother_id]
                child = self.__gen_child(father, mother)
                child[1] = self.current_gen
                child[2] = i
                childs.append(child)

        return childs


    def __random_mutate_elem(self, elem, p=0.5):
        """
            elem: keras element description
        """
        mutate = np.random.choice([True, False], p=[p, 1.0 - p])

        if mutate:
            new_elem = copy.deepcopy(elem)
            old_params = new_elem['params']

            random_key = random.choice(list(old_params.keys()))
            
            val = old_params[random_key]

            mutated_param = self.__random_mutate_param(random_key, val)
            new_elem['params'][random_key] = mutated_param

            return new_elem
        else:
            return elem

    def __random_mutate_param(self, key, value):
        if key in ['activation', 'Activation']:
            return random.choice(self.ACTIVATIONS)
        elif key in ['units','Units']:
            # integer value
            mean = value
            stddev = np.sqrt(value)

            if stddev < 2:
                stddev = 2

            new_val = int(np.random.normal(loc=mean, scale=stddev) )
            
            if(new_val < 2):
                return 2
            else:
                return new_val
        else:
            # all floating values
            mean = value
            stddev = np.sqrt(value)

            if stddev < 0.1:
                stddev = 0.1

            new_val = np.random.normal(loc=mean, scale=stddev)

            if new_val < 0.0:
                return 0.0
            else:
                return new_val

    

    def __random_mutate(self, population, p_mut=0.3, p_add=0.1, p_rem=0.1):

        # mutate params
        for i in range(len(population) ):

            mutate = np.random.choice([True, False], p=[p_mut, 1.0 - p_mut])

            if mutate:
                # mutate existing params
                net_old = population[i][0]['net']

                net_new = [net_old[0]]

                for j in range(len(net_old)-2):
                    net_elem = net_old[j+1]                    
                    new_elem = self.__random_mutate_elem(net_elem, p=0.5)
                    net_new.append(new_elem)

                net_new.append(net_old[-1])
                population[i][0]['net'] = net_new

            # ADD LAYERS RANDOMLY
            add = np.random.choice([True, False], p=[p_add, 1.0 - p_add])

            if add:
                
                # add layer
                net_old = population[i][0]['net']

                # randomly insert layer
                # layer: dense
                # units: calc mean and std dev of existing layers
                # position: random int 

                units = []

                num_dynamic_layers = len(net_old)-2
                num_dyn_param_layer = int(num_dynamic_layers / 2)

                for j in range(num_dynamic_layers):
                    elem = net_old[j+1]
                    if 'units' in elem['params']:
                        units.append(elem['params']['units'])

                units = np.array(units)
                mean = np.mean(units)
                stddev = np.std(units)

                random_parameter_layer = {
                    'type':'dense',
                    'params': {
                        'units': int( np.random.normal(mean, stddev) ),
                        'activation': random.choice(self.ACTIVATIONS)
                    }
                }

                random_constant_layer = {
                    'type':'dropout',
                    'params':{
                        'rate': np.random.random_sample()
                    }
                }

                pos = np.random.randint(0, high=num_dyn_param_layer+1)
                # print('add layer to pos %d' % pos)

                pos = pos * 2 + 1

                net_old.insert(pos, random_constant_layer)
                net_old.insert(pos, random_parameter_layer)

                population[i][0]['net'] = net_old

            rem = np.random.choice([True, False], p=[p_rem, 1.0 - p_rem])
            if rem:
                net_old = population[i][0]['net']

                num_dynamic_layers = len(net_old)-2
                num_dyn_param_layer = int(num_dynamic_layers / 2)

                if num_dyn_param_layer > 0:
                    rem_id = np.random.randint(0, high=num_dyn_param_layer)
                    rem_id_global = rem_id * 2 + 1
                    # remove dynamic layer
                    del population[i][0]['net'][rem_id_global]
                    del population[i][0]['net'][rem_id_global]

        return population
        
    def __gen_string(self, pop_entry):
        return 'Model: Generation %d, Id %d, Fitness %f, Val Acc: %f' % (pop_entry[1], pop_entry[2], pop_entry[3], pop_entry[4])

    def __gen_string_descr(self, pop_entry):
        out_str = ''

        net = pop_entry[0]['net']
        for i,elem in enumerate(net):
            if elem['type'] in ['input','Input']:
                out_str += '%s, ' % elem['type']
            elif elem['type'] in ['dense', 'Dense']:
                out_str += '%s (%d, %s), ' % (elem['type'], elem['params']['units'], elem['params']['activation'])
            elif elem['type'] in ['dropout','Dropout']:
                out_str += '%s (%f), ' % (elem['type'], elem['params']['rate'])

        return out_str

    def __print_pop(self, population):
        for i,elem in enumerate(population):
            print('%s --- %s' % (self.__gen_string(elem), self.__gen_string_descr(elem) ) )

    def evolve(self, X, y, generations = 10):

        self.current_gen = 0

        for i in range(generations):
            self.current_gen += 1
            print('Generation: %d, Population Size: %d' % (self.current_gen, len(self.model_population)))

            # print('\t\tCurrent Population:')
            # self.__print_pop(self.model_population)

            # UPDATE FITNESSES

            with tqdm(total=len(self.model_population), ascii=True, dynamic_ncols=True, desc='-- update fitnesses...') as pbar:
                # pbar.set_description('\t\t')
                self.__update_fitnesses(X,y, pbar)

            # RANK POPULATION
            print('-- rank population...')
            self.__rank_population()
            print('\tBest Model: %s' % (self.__gen_string(self.model_population[0])) )
            
            # LET THEM ALL DIE
            print('-- die hard')
            num_deaths = self.__die_hard()
            print('\tnumber of deaths: %d' % num_deaths)

            # GENERATE CHILDS
            print('-- generate childs...')
            childs = self.__gen_childs(self.model_population, n_first_add=5, n_childs=self.n_childs_per_gen)

            # MUTATE CHILDS
            print('-- random mutate childs...')
            mutated_childs = self.__random_mutate(childs, p_mut = self.p_mutate, p_add=self.p_add_layer, p_rem=self.p_del_layer)

            # APPEND CHILDS TO POPULATION
            print('-- give birth to childs...')
            self.model_population.extend(mutated_childs)

            print('')

        
        self.__rank_population()
        return self.__build_model(self.model_population[0][0])


        # model = self.__build_model(self.model_descr)

        # # compute fitnesses

        

        # # performance
        # fitness = self.__compute_fitness(losses, accs, val_losses, val_accs)


        # plt.plot(losses, label='loss')
        # plt.plot(accs, label='acc')
        # plt.plot(val_losses, label='val loss')
        # plt.plot(val_accs, label='val acc')
        # plt.plot(np.ones(len(losses)) * fitness, label='fitness')
        # plt.legend()
        # plt.show()


    def fit(self,X,y):
        '''
            growing NN fit
        '''
        model = self.__build_model(self.model_descr)
        model.summary()
        hist = model.fit(X, y, epochs=10)
        return model


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    print('Growing NN test')

    num_samples = 1000

    gnn = GrowingNN(n_childs_per_gen=10, max_population=15, p_add_layer=0.5, p_del_layer=0.5, p_mutate=0.5)
    
    solvable = True

    if solvable:
        # solvable example
        X,y = make_classification(n_samples=num_samples, n_features=200)
    else:
        
        # unsolvable
        X = np.random.random_sample(size=(num_samples,200))
        y = np.random.random_sample(size=num_samples)

    gnn.evolve(X,y)

