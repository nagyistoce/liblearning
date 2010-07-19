/****************************************************************************
**
** Copyright (C) 2009-2010 TECHNOGERMA Systems France and/or its subsidiary(-ies).
** Contact: Technogerma Systems France Information (contact@technogerma.fr)
**
** This file is part of the CAMP library.
**
** CAMP is free software: you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** CAMP is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
** 
** You should have received a copy of the GNU Lesser General Public License
** along with CAMP.  If not, see <http://www.gnu.org/licenses/>.
**
****************************************************************************/


#include <camp/detail/observernotifier.hpp>
#include <camp/observer.hpp>
#include <cassert>


namespace camp
{
namespace detail
{
//-------------------------------------------------------------------------------------------------
void ObserverNotifier::addObserver(Observer* observer)
{
    assert(observer != 0);

    m_observers.insert(observer);
}

//-------------------------------------------------------------------------------------------------
void ObserverNotifier::removeObserver(Observer* observer)
{
    m_observers.erase(observer);
}

//-------------------------------------------------------------------------------------------------
ObserverNotifier::ObserverNotifier()
{
}

//-------------------------------------------------------------------------------------------------
ObserverNotifier::ObserverIterator ObserverNotifier::observersBegin() const
{
    return m_observers.begin();
}

//-------------------------------------------------------------------------------------------------
ObserverNotifier::ObserverIterator ObserverNotifier::observersEnd() const
{
    return m_observers.end();
}

} // namespace detail

} // namespace camp
