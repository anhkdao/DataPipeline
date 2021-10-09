summary: Final Presentation
id: App-Deployment
categories: Sample
tags: medium
status: Published
authors: anhdao

# Deployment

<!-- ------------------------ -->
## Setting up the Environment
Duration: 5

### Steps by steps process I performed to make Nowcast dataset

- Linux, MacOs or windows etc

- Python 3.8 and above

- Streamlit library

- H5py

- Pandas

- Numpy as a dependency

- Hosting service : any one that offers Virtual private server(VPS)

<!-- ------------------------ -->
## Network Specification
Duration: 5

![alt-text-here](assets/network\_specification.png)

<!-- ------------------------ -->
## In Depths
Duration: 5

### There are 2 main steps

- To develop the program needed for this data evaluation, one will require an operating system that fully supports the latest version of python.  In this particular project, streamlit, pandas, numpy and h5py was being utilized.


- Streamlit enables the  rendering of the dataset for viewing on the browser by creating a port and binding of the system ip, it basically serves as a server and a visualiser for the data. It contains needed widgets for displaying in tabular form, line charts amongst others.

- Pandas is used for reading the data files that are in csv and it makes use of python numpy library in its internals

<!-- ------------------------ -->
## In Depths Continued
Duration: 5

- Now to make it accessible for the entire internet to view, I made use of a VPS which makes it easy to configure, self host and manage the program, for this particular service I made use of cloud service platforms, in my case I made it available on Amazon web service and Azure to host, to ensure optimal uptime

- On Azure, I depoyed a minimum of 16gib ram and 32gib hard disk.


<!-- ------------------------ -->
## Screenshot
Duration: 5

![alt-text-here](assets/graphs.png)




