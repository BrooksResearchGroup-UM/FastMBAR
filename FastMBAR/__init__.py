"""
FastMBAR is an implementation of the multistate Bennette acceprance ratio 
(MBAR) [1] method using the PyTorch [2] library. Comparing with the package 
pymbar [3], FastMBAR is faster when calculating free energyies for a large
 num of states with a large num of conformations.
"""

from .fastmbar import FastMBAR
__all__ = ['FastMBAR']